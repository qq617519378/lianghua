# core_utils.py
import logging
import threading
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import config

# 从config导入APP_NAME和RAW_CACHE_MAX_HOURS
from config import APP_NAME, RAW_CACHE_MAX_HOURS, SIGNAL_THRESHOLD_DP_A2, \
    SIGNAL_THRESHOLD_DP_A201, SIGNAL_THRESHOLD_LR_A201

class RollingFeatureCache:
    """缓存已合并的 3min 特征和衍生指标，支持增量 append。"""
    def __init__(self, max_hours: int):
        from datetime import timezone, timedelta
        import pandas as pd
        import threading

        self.logger = logging.getLogger(f"{APP_NAME}.RollingFeatureCache")
        self.max_hours = max_hours

        # 初始化一个索引名叫 time_window 的空 DataFrame
        self.df = pd.DataFrame()
        self.df.index.name = 'time_window'
        self._lock = threading.RLock()
        self.logger.info(f"RollingFeatureCache 初始化，最长缓存时长：{max_hours} 小时")

    def initialize(self, full_df: pd.DataFrame):
        """首次全量加载，用历史所有行初始化缓存。"""
        with self._lock:
            self.df = full_df.copy()

    def append(self, new_df: pd.DataFrame):
        """增量追加并修剪超出 max_hours 的旧数据。"""
        if new_df.empty: return
        with self._lock:
            combined = pd.concat([self.df, new_df])
            combined = combined[~combined.index.duplicated(keep='last')].sort_index()
            cutoff = pd.Timestamp.now(tz=pd.Timestamp.utcnow().tz) - pd.Timedelta(hours=self.max_hours)
            self.df = combined[combined.index >= cutoff]

    def get_copy(self) -> pd.DataFrame:
        with self._lock:
            return self.df.copy()

    def latest_ts(self) -> Optional[pd.Timestamp]:
        with self._lock:
            return self.df.index.max() if not self.df.empty else None

class EMA:
    """指数移动平均 (EMA) 类。"""

    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = decay
        self.shadow_params = {
            name: param.detach().clone().float()
            for name, param in model.state_dict().items()
            if param.dtype.is_floating_point
        }
        self.logger = logging.getLogger(f"{APP_NAME}.EMA")  # 使用config中的APP_NAME
        self.logger.debug(f"EMA init, decay={decay}, model {type(model).__name__}, {len(self.shadow_params)} params.")

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.state_dict().items():
            if name in self.shadow_params and param.dtype.is_floating_point:
                self.shadow_params[name].mul_(self.decay).add_(param.float(), alpha=1 - self.decay)

    def apply_shadow(self, model: nn.Module):
        model_state_dict = model.state_dict()
        for name, shadow_value in self.shadow_params.items():
            if name in model_state_dict:
                model_state_dict[name].copy_(shadow_value)
        self.logger.debug(f"EMA shadow weights applied to model {type(model).__name__}.")
    def load_shadow(self, shadow_params: Dict[str, torch.Tensor]):
        """
        从外部直接加载一组 EMA 影子参数字典，
        以便 apply_shadow 时生效。
        """
        self.shadow_params = shadow_params



class RollingRawCache:
    """滚动原始数据缓存类。"""

    def __init__(self, max_hours: int = RAW_CACHE_MAX_HOURS):  # 使用config中的RAW_CACHE_MAX_HOURS
        self.logger = logging.getLogger(f"{APP_NAME}.RollingRawCache")
        self.max_hours = max_hours
        self.df_market: pd.DataFrame = pd.DataFrame(columns=['timestamp']).set_index('timestamp')
        self.df_chain: pd.DataFrame = pd.DataFrame(columns=['timestamp']).set_index('timestamp')
        self.df_market.index = pd.to_datetime(self.df_market.index, utc=True)
        self.df_chain.index = pd.to_datetime(self.df_chain.index, utc=True)
        self._lock = threading.RLock()
        self.logger.info(f"滚动缓存初始化，最长缓存时长：{max_hours} 小时")

    def _append_and_trim(self, existing_df: pd.DataFrame, new_chunk: pd.DataFrame) -> pd.DataFrame:
        if new_chunk.empty: return existing_df
        if not isinstance(new_chunk.index, pd.DatetimeIndex):
            self.logger.error(f"New chunk index type error: {type(new_chunk.index)}.");
            return existing_df
        if new_chunk.index.tz is None:
            try:
                new_chunk.index = new_chunk.index.tz_localize('UTC')
            except Exception as e:
                self.logger.error(f"Index tz_localize failed: {e}"); return existing_df
        elif new_chunk.index.tz.utcoffset(None) != timedelta(0):
            try:
                new_chunk.index = new_chunk.index.tz_convert('UTC')
            except Exception as e:
                self.logger.error(f"Index tz_convert failed: {e}"); return existing_df

        combined_df = pd.concat([existing_df, new_chunk])
        df_out = combined_df[~combined_df.index.duplicated(keep='last')].sort_index()
        cutoff_time = pd.Timestamp.now(tz=timezone.utc) - pd.Timedelta(hours=self.max_hours)
        df_out = df_out[df_out.index >= cutoff_time]
        return df_out

    def append_market_data(self, new_chunk: pd.DataFrame):
        with self._lock: self.df_market = self._append_and_trim(self.df_market, new_chunk)

    def append_chain_data(self, new_chunk: pd.DataFrame):
        with self._lock: self.df_chain = self._append_and_trim(self.df_chain, new_chunk)

    def get_market_data_copy(self) -> pd.DataFrame:
        with self._lock: return self.df_market.copy()

    def get_chain_data_copy(self) -> pd.DataFrame:
        with self._lock: return self.df_chain.copy()

    def get_latest_market_timestamp(self) -> Optional[pd.Timestamp]:
        with self._lock: return self.df_market.index.max() if not self.df_market.empty else None

    def get_latest_chain_timestamp(self) -> Optional[pd.Timestamp]:
        with self._lock: return self.df_chain.index.max() if not self.df_chain.empty else None


class SignalGenerator:
    """信号生成器。"""

    def __init__(self):
        # 修改日志记录器名称，避免与策略服务冲突
        self.logger = logging.getLogger(f"{config.APP_NAME}.核心工具.信号生成器")
        self.logger.info("信号生成器已初始化.")

    def generate_signal_from_a2(self, pred_dp_a2: Optional[float]) -> Dict[str, Any]:
        s = {"source": "model_a2_30s", "type": "NEUTRAL", "strength": 0.0, "raw_pred_dprice": pred_dp_a2, "reason": ""}
        if pred_dp_a2 is None: s["reason"] = "无预测"; return s

        # 使用 config.py 中的阈值
        threshold_dp_a2 = getattr(config, 'SIGNAL_THRESHOLD_DP_A2', 20.0)  # 从config获取，带默认值

        if pred_dp_a2 > threshold_dp_a2:
            s.update({"type": "BUY", "strength": min(1.0, pred_dp_a2 / (threshold_dp_a2 * 3))})
        elif pred_dp_a2 < -threshold_dp_a2:
            s.update({"type": "SELL", "strength": min(1.0, abs(pred_dp_a2) / (threshold_dp_a2 * 3))})
        else:
            reason_dprice = f"{pred_dp_a2:.2f}" if pred_dp_a2 is not None else "N/A"
            s["reason"] = f"ΔPrice({reason_dprice}) 未达到阈值({threshold_dp_a2})"
        self.logger.debug(f"A2 模型独立信号: {s}");
        return s

    def generate_signal_from_a201(self, preds_a201: Optional[Tuple[float, float]]) -> Dict[str, Any]:
        s = {"source": "model_a201_3min", "type": "NEUTRAL", "strength": 0.0, "raw_pred_logret": None,
             "raw_pred_dprice": None, "reason": ""}
        if preds_a201 is None or not isinstance(preds_a201, tuple) or len(preds_a201) != 2:
            s["reason"] = "无预测或格式错误";
            return s

        plr, pdp = preds_a201
        s.update({"raw_pred_logret": plr, "raw_pred_dprice": pdp})

        # 从config获取阈值
        threshold_lr_a201 = getattr(config, 'SIGNAL_THRESHOLD_LR_A201', 0.0003)
        threshold_dp_a201 = getattr(config, 'SIGNAL_THRESHOLD_DP_A201', 50.0)

        lrb, lrs = plr > threshold_lr_a201, plr < -threshold_lr_a201
        dpb, dps = pdp > threshold_dp_a201, pdp < -threshold_dp_a201

        if lrb and dpb:
            s.update({"type": "BUY", "reason": "对数收益率和ΔPrice均看涨", "strength": (min(1.0, plr / (
                    threshold_lr_a201 * 5)) + min(1.0, pdp / (threshold_dp_a201 * 3))) / 2})
        elif lrs and dps:
            s.update({"type": "SELL", "reason": "对数收益率和ΔPrice均看跌", "strength": (min(1.0, abs(plr) / (
                    threshold_lr_a201 * 5)) + min(1.0, abs(pdp) / (threshold_dp_a201 * 3))) / 2})
        elif lrb and not dps:  # 对数收益率看涨，ΔPrice不强烈反对 (即不强烈看跌)
            s.update({"type": "WEAK_BUY", "reason": "对数收益率看涨, ΔPrice未强烈反对",
                      "strength": min(1.0, plr / (threshold_lr_a201 * 5)) * 0.5})
        elif lrs and not dpb:  # 对数收益率看跌，ΔPrice不强烈反对 (即不强烈看涨)
            s.update({"type": "WEAK_SELL", "reason": "对数收益率看跌, ΔPrice未强烈反对",
                      "strength": min(1.0, abs(plr) / (threshold_lr_a201 * 5)) * 0.5})
        else:
            reason_plr = f"{plr:.5f}" if plr is not None else "N/A"
            reason_pdp = f"{pdp:.2f}" if pdp is not None else "N/A"
            s["reason"] = f"对数收益率({reason_plr})或ΔPrice({reason_pdp})信号不足或冲突"
        self.logger.debug(f"A201 模型独立信号: {s}");
        return s

    def combine_signals(self, sig_a2: Dict[str, Any], sig_a201: Dict[str, Any],
                        w_a2: float = 0.4, w_a201: float = 0.6,
                        time_decay_tau_seconds: Optional[float] = None) -> Dict[str, Any]:
        """
        合并来自不同模型的信号。
        新增: 支持时间衰减 (如果提供了 tau 和信号时间戳)。
        新增: 冲突时可输出 HEDGE 信号。
        """
        final_signal = {"type": "NEUTRAL", "strength": 0.0, "reason": "初始状态", "contributing_signals": []}

        signals_to_process = []
        for sig_dict, base_weight, model_key in [(sig_a2, w_a2, "a2"), (sig_a201, w_a201, "a201")]:
            if sig_dict["type"] != "NEUTRAL":
                weight = base_weight
                # --- 实验性：时间衰减 ---
                # if time_decay_tau_seconds and sig_dict.get("timestamp"): # 假设信号字典中有时间戳
                #     age_seconds = (pd.Timestamp.now(tz=timezone.utc) - sig_dict["timestamp"]).total_seconds()
                #     decay_factor = np.exp(-age_seconds / time_decay_tau_seconds)
                #     weight *= decay_factor
                #     sig_dict["time_decayed_weight"] = weight # 记录一下
                signals_to_process.append({"signal_data": sig_dict, "weight": weight})

        if not signals_to_process:
            final_signal["reason"] = "所有模型信号均为中性"
            self.logger.info(
                f"合并信号: {final_signal['type']}, 强度: {final_signal['strength']:.3f}, 原因: '{final_signal['reason']}'");
            return final_signal

        final_signal["contributing_signals"] = [s["signal_data"] for s in signals_to_process]

        if len(signals_to_process) == 1:
            s_info = signals_to_process[0]["signal_data"]
            final_signal.update({"type": s_info["type"], "strength": s_info["strength"],
                                 "reason": f"仅来自 {s_info['source']} 的信号"})
            self.logger.info(
                f"合并信号: {final_signal['type']}, 强度: {final_signal['strength']:.3f}, 原因: '{final_signal['reason']}'");
            return final_signal

        # 处理两个模型都有非中性信号的情况
        s1_info, w1 = signals_to_process[0]["signal_data"], signals_to_process[0]["weight"]
        s2_info, w2 = signals_to_process[1]["signal_data"], signals_to_process[1]["weight"]

        # 标准化信号方向 (BUY, SELL)，忽略 WEAK_ 前缀
        s1_dir = s1_info["type"].replace("WEAK_", "")
        s2_dir = s2_info["type"].replace("WEAK_", "")

        if s1_dir == s2_dir:  # 方向一致
            # 加权平均强度，但如果一个是WEAK而另一个不是，可能需要调整逻辑
            combined_strength = (s1_info["strength"] * w1 + s2_info["strength"] * w2) / (w1 + w2)
            final_signal_type = s1_dir  # 因为方向一致
            # 如果一个是WEAK，一个是强信号，最终信号类型可能是强的那个，或者也是WEAK但强度更高
            if "WEAK_" in s1_info["type"] and "WEAK_" not in s2_info["type"]:
                final_signal_type = s2_dir  # 取强信号类型
            elif "WEAK_" not in s1_info["type"] and "WEAK_" in s2_info["type"]:
                final_signal_type = s1_dir  # 取强信号类型
            elif "WEAK_" in s1_info["type"] and "WEAK_" in s2_info["type"]:
                final_signal_type = f"WEAK_{s1_dir}"

            final_signal.update({"type": final_signal_type, "strength": combined_strength,
                                 "reason": f"{s1_info['source']}({s1_info['type']}) 与 {s2_info['source']}({s2_info['type']}) 方向一致 ({s1_dir})"})
        else:  # 方向冲突
            # 方案1: 抵消，看谁的加权强度大
            # weighted_strength_s1 = s1_info["strength"] * w1 if s1_dir == "BUY" else -s1_info["strength"] * w1
            # weighted_strength_s2 = s2_info["strength"] * w2 if s2_dir == "BUY" else -s2_info["strength"] * w2
            # net_strength_numeric = weighted_strength_s1 + weighted_strength_s2
            # final_signal_strength_abs = abs(net_strength_numeric)
            # if net_strength_numeric > 0.01: # 留一点死区
            #     final_signal.update({"type": "BUY", "strength": final_signal_strength_abs, "reason": "冲突但买方胜出"})
            # elif net_strength_numeric < -0.01:
            #     final_signal.update({"type": "SELL", "strength": final_signal_strength_abs, "reason": "冲突但卖方胜出"})
            # else: # 强度接近，判定为对冲或中性
            #     final_signal.update({"type": "HEDGE", "strength": 0.0, "reason": f"信号冲突且强度接近: {s1_info['source']}({s1_info['type']}) vs {s2_info['source']}({s2_info['type']})"})

            # 方案2: 直接标记为冲突/对冲 (更保守)
            final_signal.update({"type": "HEDGE", "strength": 0.0,  # 或者是一个小的固定强度表示对冲意愿
                                 "reason": f"信号冲突: {s1_info['source']}({s1_info['type']}) vs {s2_info['source']}({s2_info['type']}). 建议对冲或观望."})
            # 也可以根据哪个信号更“强”来决定对冲的方向，例如，如果BUY信号强度远大于SELL，HEDGE信号可以偏向减少空头敞口
            # 这里简化为中性对冲

        final_signal["strength"] = min(1.0, max(0.0, final_signal["strength"]))  # 确保强度在0-1
        self.logger.info(
            f"合并信号: {final_signal['type']}, 强度: {final_signal['strength']:.3f}, 原因: '{final_signal['reason']}'");
        return final_signal
