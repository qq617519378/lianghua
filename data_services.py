# data_services.py (修正并中文化)
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import hashlib
import joblib
from datetime import timezone, timedelta  # <--- 确保 timedelta 也导入

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text, Engine

# 从config导入相关配置
import config

# 从core_utils导入

# 从外部模块导入
from a1 import NewDataProcessor as DataProcessorA1
from a2 import HybridNeuralODEModel as ModelA2
from a201 import PricePredictorModel as ModelA201, dprice_decode as dprice_decode_a201
from core_utils import RollingRawCache, EMA
from a1 import CRYPTO_AGG_RULES, BC_COLUMNS
from a101 import compute_features
from a101 import resample_and_merge, generate_log_return_target, add_features, normalize_features


class DataPipelineManager:
    def __init__(self, raw_cache_ref: RollingRawCache, db_engine: Engine):
        self.logger = logging.getLogger(f"{config.APP_NAME}.DataPipelineManager")
        self.raw_cache = raw_cache_ref
        self.engine = db_engine
        end_ts = pd.Timestamp.now(tz=timezone.utc)
        start_ts = end_ts - pd.Timedelta(config.MIN_CHAIN_DATA_HISTORY_FOR_A101_STR)

        # 初始化 chain_full_cache
        try:
            df_chain_hist = pd.read_sql(
                text(f"SELECT * FROM {config.DB_CHAIN_FEATURES_TABLE} "
                     "WHERE timestamp >= :s AND timestamp <= :e ORDER BY timestamp"),  # 添加 ORDER BY
                self.engine,
                params={
                    "s": start_ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "e": end_ts.strftime("%Y-%m-%d %H:%M:%S")
                },
                index_col="timestamp", parse_dates=["timestamp"]
            )
            if not df_chain_hist.empty:
                if df_chain_hist.index.tz is None:
                    df_chain_hist.index = df_chain_hist.index.tz_localize("UTC")
                else:
                    df_chain_hist.index = df_chain_hist.index.tz_convert("UTC")
                # 去重，保留最新的
                df_chain_hist = df_chain_hist[~df_chain_hist.index.duplicated(keep='last')]
            self.raw_cache.append_chain_data(df_chain_hist)  # 先添加到raw_cache
            self.chain_full_cache = self.raw_cache.get_chain_data_copy()  # 然后从raw_cache获取作为初始全量
            self.logger.info(f"初始化链上数据缓存 (chain_full_cache)，共 {len(self.chain_full_cache)} 条记录。")
        except Exception as e_init_chain:
            self.logger.error(f"初始化链上数据缓存失败: {e_init_chain}", exc_info=True)
            self.chain_full_cache = pd.DataFrame(columns=BC_COLUMNS).set_index(
                pd.to_datetime([]).tz_localize('UTC'))  # 创建带正确时区的空DataFrame
        try:
            k_interval_min = int(pd.Timedelta(config.A1_FREQ).total_seconds() / 60)  # 30 s → 0.5 min
            minutes_a2_needed = config.A2_SEQ_LEN * k_interval_min
            hist_minutes_market = max(
                minutes_a2_needed,
                config.MIN_HISTORY_MINUTES_A2
            ) + 120  # 再额外多留两小时

            end_ts_mkt = pd.Timestamp.now(tz=timezone.utc)
            start_ts_mkt = end_ts_mkt - pd.Timedelta(minutes=hist_minutes_market)

            q_sql = text(
                f"""
                        SELECT *
                        FROM {config.DB_MARKET_DATA_TABLE}
                        WHERE time_window >= :s AND time_window <= :e
                        ORDER BY time_window
                        """
            )
            df_mkt_hist = pd.read_sql(
                q_sql, self.engine,
                params={"s": start_ts_mkt.strftime("%Y-%m-%d %H:%M:%S"),
                        "e": end_ts_mkt.strftime("%Y-%m-%d %H:%M:%S")},
                index_col="time_window", parse_dates=["time_window"]
            )
            if not df_mkt_hist.empty:
                if df_mkt_hist.index.tz is None:
                    df_mkt_hist.index = df_mkt_hist.index.tz_localize(timezone.utc)
                else:
                    df_mkt_hist.index = df_mkt_hist.index.tz_convert(timezone.utc)
                self.raw_cache.append_market_data(df_mkt_hist)
                self.logger.info(
                    f"初始化行情数据缓存，共 {len(df_mkt_hist)} 条记录，跨度 {hist_minutes_market} min。")
        except Exception as e_hist:
            self.logger.error(f"初始化行情数据缓存失败: {e_hist}", exc_info=True)

        self.processor_a1 = DataProcessorA1(conn_str=None, freq=config.A1_FREQ,
                                            db_engine_override=self.engine)
        self.processor_a1.scaler_path = config.ARTIFACTS_DIR / config.A1_SCALER_FILENAME

        self.features_a2: Optional[List[str]] = None
        self.features_a201: Optional[List[str]] = None

        self.seq_len_a201 = config.A201_SEQ_LEN
        self.freq_minutes_a201 = int(pd.Timedelta(config.A101_AGG_PERIOD).total_seconds() / 60)
        self.seq_len_a2 = config.A2_SEQ_LEN

        self.target_scaler_a2: Optional[StandardScaler] = None
        self.scaler_a2_target_path = config.ARTIFACTS_DIR / f"target_scaler_a2_{config.A1_FREQ}_h{config.A1_HORIZON_PERIODS_FOR_A2}.joblib"
        if self.scaler_a2_target_path.exists():
            try:
                self.target_scaler_a2 = joblib.load(self.scaler_a2_target_path)
                self.logger.info(f"成功加载A2目标缩放器: {self.scaler_a2_target_path}")
            except Exception as e:
                self.logger.warning(f"加载 A2 目标缩放器失败: {e}")

        k_interval_min = int(pd.Timedelta(config.A1_FREQ).total_seconds() / 60)
        minutes_a2_needed = config.A2_SEQ_LEN * k_interval_min

        self.buffer_minutes = max(
            config.MIN_HISTORY_MINUTES_A2,
            minutes_a2_needed,
            self.freq_minutes_a201 * config.A201_SEQ_LEN,
            2 * config.A101_ROLL_WINDOW * int(pd.Timedelta(config.A101_AGG_PERIOD).total_seconds() / 60)
        ) + 120
        self.logger.info(f"A2/A201 缓冲窗口设置为 {self.buffer_minutes} min")

        # 初始化 market_buffer 和 chain_buffer (在update_caches之前可能是空的)
        df_raw_m = self.raw_cache.get_market_data_copy()
        df_raw_c = self.raw_cache.get_chain_data_copy()  # 注意：这里是raw_cache中的链上数据，可能不是全量的
        self.market_buffer = df_raw_m.last(f"{self.buffer_minutes}min") if not df_raw_m.empty else df_raw_m
        self.chain_buffer = df_raw_c.last(f"{self.buffer_minutes}min") if not df_raw_c.empty else df_raw_c

        self.a1_horizon_seconds_for_a2_target = int(
            pd.Timedelta(config.A1_FREQ).total_seconds() * config.A1_HORIZON_PERIODS_FOR_A2)
        self.a201_horizon_seconds = int(
            pd.Timedelta(config.A101_AGG_PERIOD).total_seconds() * config.A101_PRED_HORIZON_K)
        self.logger.info("DataPipelineManager 数据管道管理器已初始化。")

    def get_latest_processed_data_for_a2(self) -> Optional[pd.DataFrame]:
        """
        增量生成 A2 所需 30 s 频率特征。
        返回最近 seq_len_a2 行已标准化特征；不足则返回 None。
        """
        # 1) 缓存刷新 (确保 raw_cache 是最新的)
        self.update_caches()

        # 2) 提取 buffer, 为 a1.py 增加额外的 lookback 以处理初始 NaN
        a1_lookback_buffer_str = getattr(config, 'A1_PROCESSOR_ROLLING_BUFFER_ESTIMATE_STR', "0m")  # 从config获取，若无则为0分钟
        try:
            a1_lookback_buffer_td = pd.Timedelta(a1_lookback_buffer_str)
        except ValueError:
            self.logger.warning(
                f"配置的 A1_PROCESSOR_ROLLING_BUFFER_ESTIMATE_STR ('{a1_lookback_buffer_str}') 无效，lookback buffer 设为0。")
            a1_lookback_buffer_td = pd.Timedelta(minutes=0)

        # 计算 market_buffer 的起始时间点
        # market_buffer 本身已经包含了 buffer_minutes 的数据
        # 我们要确保传递给 a1.py 的 df_m 是从 market_buffer 的起点再往前推 a1_lookback_buffer_td
        # 这意味着 raw_cache 需要有足够的数据

        df_raw_m_full = self.raw_cache.get_market_data_copy()
        df_raw_c_full = self.raw_cache.get_chain_data_copy()  # 或者用 self.chain_full_cache，取决于策略

        # effective_buffer_start_time = (pd.Timestamp.now(tz=timezone.utc) -
        #                                pd.Timedelta(minutes=self.buffer_minutes) -
        #                                a1_lookback_buffer_td)
        # 使用 market_buffer 的当前最早时间作为参考点，而不是 pd.Timestamp.now()
        current_market_buffer_start_ts = self.market_buffer.index.min() if not self.market_buffer.empty else pd.Timestamp.now(
            tz=timezone.utc) - pd.Timedelta(minutes=self.buffer_minutes)
        effective_df_m_start_ts = current_market_buffer_start_ts - a1_lookback_buffer_td

        df_m = df_raw_m_full[df_raw_m_full.index >= effective_df_m_start_ts].copy()
        # 对于链上数据，通常它的更新频率较低，使用 self.chain_buffer (已基于 buffer_minutes 截取) 即可
        # 如果需要更长的链上历史给a1，则从 df_raw_c_full 或 self.chain_full_cache 截取
        df_c = self.chain_buffer.copy()  # 或者 self.chain_full_cache.copy() 如果a1需要更长的链上历史

        if df_m.empty:
            self.logger.warning("A2 增量处理：应用 lookback buffer 后，行情数据 (df_m) 为空。")
            return None

        # 3) 统一成 UTC 时区 (已在 raw_cache 中处理，但再次确认)
        for _df_name, _df_ref in [("df_m", df_m), ("df_c", df_c)]:
            if not _df_ref.empty:
                if _df_ref.index.tz is None:
                    _df_ref.index = _df_ref.index.tz_localize("UTC")
                elif _df_ref.index.tz.utcoffset(None) != timedelta(0):  # 检查是否真的是UTC
                    _df_ref.index = _df_ref.index.tz_convert("UTC")

        # 4) A1 预处理
        try:
            proc_df_from_a1 = self.processor_a1.get_processed_data_from_df(df_m, df_c)
        except Exception as e:
            self.logger.error(f"A2 增量处理：A1 管道失败: {e}", exc_info=True)
            return None

        if proc_df_from_a1.empty:
            self.logger.warning("A2 增量处理：a1.py 处理后 DataFrame 为空。")
            return None
        self.logger.info(f"A2 增量处理：a1.py 处理后得到 {len(proc_df_from_a1)} 行数据。")

        # 5) 获取A2模型期望的特征列表
        expected_model_features = self.features_a2
        if not expected_model_features:
            self.logger.error("A2 增量处理：DataPipelineManager.features_a2 未被设置。")
            return None

        # 6) 特征对齐和填充
        # 6.1) 识别 proc_df_from_a1 中用于目标生成的辅助列和模型特征列
        # 辅助列（如 'kline_close'）也可能被用作模型特征，所以 expected_model_features 是主要的特征集合

        # 从 proc_df_from_a1 中提取出模型需要的特征列，以及目标生成需要的 'kline_close'
        # 确保 'kline_close' 存在于 proc_df_from_a1
        if 'kline_close' not in proc_df_from_a1.columns:
            self.logger.error("A2 增量处理: a1.py输出中缺少 'kline_close' 列，无法继续。")
            return None

        # 创建一个包含所有期望特征和 'kline_close' 的列清单（去重）
        cols_to_work_with_set = set(expected_model_features)
        cols_to_work_with_set.add('kline_close')  # 确保kline_close在其中
        # 如果 proc_df_from_a1 有索引名且不冲突，也保留
        if proc_df_from_a1.index.name and proc_df_from_a1.index.name not in cols_to_work_with_set:
            # 通常索引名如 time_window 会在 get_processed_data_from_df 后变成普通列或消失
            pass

        cols_to_work_with_list = list(cols_to_work_with_set)

        # 筛选 proc_df_from_a1，只保留这些工作列，并处理缺失
        # 用 reindex 一次性取出所有目标列，并用 0.0 填充所有缺失
        working_df = proc_df_from_a1.reindex(columns=cols_to_work_with_list)
        working_df = working_df.fillna(0.0)

        # 6.2) 对 working_df 中的模型特征列进行彻底的NaN填充
        # (此时 'kline_close' 也包含在 working_df 中，如果它也是特征之一，也会被填充)
        # 如果 'kline_close' 仅用于目标生成而非模型输入，确保它不在此阶段被错误填充（通常不会是NaN）
        features_in_working_df = [f for f in expected_model_features if f in working_df.columns]  # 确保只处理实际存在的期望特征
        working_df[features_in_working_df] = working_df[features_in_working_df].ffill().bfill().fillna(0.0)

        # 7) 生成ΔPrice目标列 (会添加到 working_df)
        # _generate_and_scale_target_for_a2 需要 'kline_close'
        working_df = self._generate_and_scale_target_for_a2(working_df)

        # 8) 确定有效数据起点 (基于模型输入特征是否都无NaN)
        # 再次检查模型特征列在 working_df 中是否干净
        if working_df[features_in_working_df].isnull().any().any():
            self.logger.error("A2 增量处理：CRITICAL! 对模型特征列填充后仍检测到NaN。这是一个逻辑错误。")
            # 强制填充作为最后手段
            working_df[features_in_working_df] = working_df[features_in_working_df].fillna(0.0)

        # dropna() 现在应该不会因为模型特征列的NaN而删除行
        df_for_model_input_check = working_df.dropna(subset=features_in_working_df, how='any')

        if df_for_model_input_check.empty:
            self.logger.warning(
                f"A2 增量处理：对齐并填充模型输入特征后，没有留下任何有效行。a1.py处理后行数: {len(proc_df_from_a1)}")
            return None

        first_valid_idx = df_for_model_input_check.index.min()
        if pd.isna(first_valid_idx):
            self.logger.warning("A2 增量处理：核心特征在dropna后导致NaT，无有效起始点。")
            return None

        # 从第一个所有模型特征都有效的行开始截取 working_df
        final_df_trimmed = working_df.loc[first_valid_idx:]
        self.logger.info(f"A2 增量处理：去除初始无效行后，剩余 {len(final_df_trimmed)} 行。")

        # 9) 检查序列长度
        if len(final_df_trimmed) < self.seq_len_a2:
            self.logger.info(
                f"A2 增量处理：数据行数 ({len(final_df_trimmed)}) 少于窗口长度 ({self.seq_len_a2})。等待更多数据。")
            return None

        self.logger.info(f"A2 增量处理：成功准备 {self.seq_len_a2} 行数据用于预测。")
        return final_df_trimmed.tail(self.seq_len_a2)

    def _fetch_new_data_from_db_to_cache(self):
        """从数据库拉取最新的行情数据到滚动缓存 (raw_cache)。"""
        now_for_query_end = pd.Timestamp.now(tz=timezone.utc)
        last_market_ts_in_cache = self.raw_cache.get_latest_market_timestamp()

        # 查询起点：如果缓存中有数据，从上次最新时间戳之后开始；否则，从 buffer_minutes + a1_lookback_buffer_td 之前开始
        if last_market_ts_in_cache:
            start_mt_s_ts = last_market_ts_in_cache + pd.Timedelta(seconds=1)  # 避免重复拉取最后一条
        else:
            a1_lookback_buffer_str = getattr(config, 'A1_PROCESSOR_ROLLING_BUFFER_ESTIMATE_STR', "0m")
            try:
                a1_lookback_buffer_td = pd.Timedelta(a1_lookback_buffer_str)
            except ValueError:
                a1_lookback_buffer_td = pd.Timedelta(minutes=0)
            start_mt_s_ts = now_for_query_end - pd.Timedelta(minutes=self.buffer_minutes) - a1_lookback_buffer_td
            self.logger.info(f"行情数据缓存为空，将从 {start_mt_s_ts} 开始拉取。")

        start_mt_s = start_mt_s_ts.strftime('%Y-%m-%d %H:%M:%S.%f')
        end_mt_s = now_for_query_end.strftime('%Y-%m-%d %H:%M:%S.%f')

        if start_mt_s_ts < now_for_query_end:
            q_market = text(
                f"""
                SELECT
                    time_window, kline_open, kline_high, kline_low, kline_close,
                    kline_volume, kline_volume AS volume,
                    spread_mean, ask_depth_mean, bid_depth_mean, imbalance_mean,
                    open_ask_price, high_ask_price, low_ask_price, close_ask_price,
                    open_bid_price, high_bid_price, low_bid_price, close_bid_price,
                    spread_min, spread_max, spread_twa,
                    ask_depth_min, ask_depth_max, ask_depth_twa,
                    bid_depth_min, bid_depth_max, bid_depth_twa,
                    imbalance_min, imbalance_max, imbalance_twa
                FROM {config.DB_MARKET_DATA_TABLE}
                WHERE time_window > :s AND time_window <= :e
                ORDER BY time_window
                """
            )
            try:
                df_new_m = pd.read_sql(q_market, self.engine, params={'s': start_mt_s, 'e': end_mt_s},
                                       index_col='time_window', parse_dates=['time_window'])
                if not df_new_m.empty:
                    if df_new_m.index.tz is None:
                        df_new_m.index = df_new_m.index.tz_localize(timezone.utc)
                    elif df_new_m.index.tz.utcoffset(None) != timedelta(0):
                        df_new_m.index = df_new_m.index.tz_convert(timezone.utc)

                    # 确保所有 CRYPTO_AGG_RULES 中的列都存在，如果不存在则补NaN
                    # 这有助于下游a1.py的resample_crypto不会因缺列报错
                    for col_name in CRYPTO_AGG_RULES.keys():
                        if col_name not in df_new_m.columns:
                            df_new_m[col_name] = np.nan
                            self.logger.debug(f"从DB加载行情数据：补齐缺失列 {col_name} 为 NaN。")
                    if 'volume' not in df_new_m.columns and 'kline_volume' in df_new_m.columns:  # a101需要'volume'
                        df_new_m['volume'] = df_new_m['kline_volume']

                    self.raw_cache.append_market_data(df_new_m)
                    self.logger.info(f"数据库→缓存：新加载 {len(df_new_m)} 条行情数据（起点 {start_mt_s}）。")
            except Exception as e:
                self.logger.error(f"数据库→缓存：行情数据加载失败: {e}", exc_info=True)
        else:
            self.logger.debug("数据库→缓存：无需加载新的行情数据 (start_mt_s_ts >= now_for_query_end)。")

    def _fetch_new_chain_data_from_db_to_cache(self):
        """从数据库拉取最新的链上特征数据到滚动缓存 (raw_cache)。"""
        now_for_query_end = pd.Timestamp.now(tz=timezone.utc)
        last_chain_ts_in_cache = self.raw_cache.get_latest_chain_timestamp()

        if last_chain_ts_in_cache:
            start_ct_ts = last_chain_ts_in_cache + pd.Timedelta(seconds=1)
        else:
            # 如果链上数据缓存为空，从一个较早的时间点开始拉取，以确保 self.chain_full_cache 能被充分填充
            start_ct_ts = now_for_query_end - pd.Timedelta(config.MIN_CHAIN_DATA_HISTORY_FOR_A101_STR)  # 使用A101需要的最少历史
            self.logger.info(f"链上数据缓存为空，将从 {start_ct_ts} 开始拉取。")

        start_ct = start_ct_ts.strftime('%Y-%m-%d %H:%M:%S.%f')
        end_ct = now_for_query_end.strftime('%Y-%m-%d %H:%M:%S.%f')

        if start_ct_ts < now_for_query_end:
            q_chain = text(
                f"SELECT * FROM {config.DB_CHAIN_FEATURES_TABLE} "
                "WHERE timestamp > :s AND timestamp <= :e ORDER BY timestamp"
            )
            try:
                df_new_c = pd.read_sql(
                    q_chain, self.engine,
                    params={'s': start_ct, 'e': end_ct},
                    index_col='timestamp', parse_dates=['timestamp']
                )
                if not df_new_c.empty:
                    if df_new_c.index.tz is None:
                        df_new_c.index = df_new_c.index.tz_localize(timezone.utc)
                    elif df_new_c.index.tz.utcoffset(None) != timedelta(0):
                        df_new_c.index = df_new_c.index.tz_convert(timezone.utc)

                    # 确保BC_COLUMNS中的列都存在
                    for col_name in BC_COLUMNS:
                        if col_name not in df_new_c.columns:
                            df_new_c[col_name] = np.nan
                            self.logger.debug(f"从DB加载链上数据：补齐缺失列 {col_name} 为 NaN。")

                    self.raw_cache.append_chain_data(df_new_c)
                    self.logger.info(f"数据库→缓存：新加载 {len(df_new_c)} 条链上特征数据（起点 {start_ct}）。")
            except Exception as e:
                self.logger.error(f"数据库→缓存：链上数据加载失败: {e}", exc_info=True)
        else:
            self.logger.debug("数据库→缓存：无需加载新的链上数据 (start_ct_ts >= now_for_query_end)。")

    def update_caches(self):
        """更新滚动缓存：先增量拉取行情数据，再增量拉取链上特征数据。然后更新滑动缓冲区。"""
        self._fetch_new_data_from_db_to_cache()
        self._fetch_new_chain_data_from_db_to_cache()

        # 更新滑动缓冲区 (market_buffer, chain_buffer)
        df_raw_m = self.raw_cache.get_market_data_copy()
        df_raw_c = self.raw_cache.get_chain_data_copy()

        current_time_for_buffer = pd.Timestamp.now(tz=timezone.utc)
        buffer_start_time = current_time_for_buffer - pd.Timedelta(minutes=self.buffer_minutes)

        self.market_buffer = df_raw_m[df_raw_m.index >= buffer_start_time].copy() if not df_raw_m.empty else df_raw_m
        self.chain_buffer = df_raw_c[df_raw_c.index >= buffer_start_time].copy() if not df_raw_c.empty else df_raw_c

        self.logger.debug(
            f"滑动缓冲区已更新。Market buffer: {len(self.market_buffer)} 行, Chain buffer: {len(self.chain_buffer)} 行。")

    def update_chain_state(self):
        """
        更新内部持有的全量链上数据缓存 (`self.chain_full_cache`)。
        """
        # self._fetch_new_chain_data_from_db_to_cache() # 确保 raw_cache 中的链上数据最新 (通常由 update_caches 调用)

        new_chain_from_raw_cache = self.raw_cache.get_chain_data_copy()

        if not new_chain_from_raw_cache.empty:
            if self.chain_full_cache.empty:
                self.chain_full_cache = new_chain_from_raw_cache
            else:
                # 仅合并新增的或更新的数据
                # combined = pd.concat([self.chain_full_cache, new_chain_from_raw_cache])
                # self.chain_full_cache = combined[~combined.index.duplicated(keep='last')].sort_index()
                # 使用 update/combine_first 避免大量重复数据（如果索引重叠很多）
                # self.chain_full_cache = self.chain_full_cache.combine_first(new_chain_from_raw_cache)
                # 为了确保新数据覆盖旧数据，且不丢失 self.chain_full_cache 中不在 new_chain_from_raw_cache 索引范围的数据：
                # 1. 从 new_chain_from_raw_cache 中取出不在 self.chain_full_cache 索引中的新行
                new_indices = new_chain_from_raw_cache.index.difference(self.chain_full_cache.index)
                # 2. 对于重叠的索引，用 new_chain_from_raw_cache 的数据更新
                self.chain_full_cache.update(new_chain_from_raw_cache)
                # 3. 添加完全新的行
                if not new_indices.empty:
                    self.chain_full_cache = pd.concat(
                        [self.chain_full_cache, new_chain_from_raw_cache.loc[new_indices]])

            self.chain_full_cache.sort_index(inplace=True)  # 确保排序
            # 修剪 self.chain_full_cache 到合理的历史长度，例如 MIN_CHAIN_DATA_HISTORY_FOR_A101_STR
            min_history_td = pd.Timedelta(config.MIN_CHAIN_DATA_HISTORY_FOR_A101_STR)
            cutoff_chain_full = pd.Timestamp.now(tz=timezone.utc) - min_history_td - pd.Timedelta(days=1)  # 再多一天buffer
            self.chain_full_cache = self.chain_full_cache[self.chain_full_cache.index >= cutoff_chain_full]

            self.logger.debug(f"链上全量缓存已更新，当前共 {len(self.chain_full_cache)} 条记录。")
        elif self.chain_full_cache.empty:
            self.logger.debug("链上全量缓存仍然为空，raw_cache中也无新数据。")
    # ======== 通用：批量读取历史行情 / 链上数据 ========
    def _load_market_from_db(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
        q = text(
            f"SELECT * FROM {config.DB_MARKET_DATA_TABLE} "
            "WHERE time_window >= :s AND time_window <= :e ORDER BY time_window"
        )
        df = pd.read_sql(q, self.engine,
                         params={"s": start_ts, "e": end_ts},
                         index_col="time_window", parse_dates=["time_window"])
        if not df.empty:
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
            # a101 需要 volume 列
            if "volume" not in df.columns and "kline_volume" in df.columns:
                df["volume"] = df["kline_volume"]
        return df

    def _load_chain_from_db(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
        q = text(
            f"SELECT * FROM {config.DB_CHAIN_FEATURES_TABLE} "
            "WHERE timestamp >= :s AND timestamp <= :e ORDER BY timestamp"
        )
        df = pd.read_sql(q, self.engine,
                         params={"s": start_ts, "e": end_ts},
                         index_col="timestamp", parse_dates=["timestamp"])
        if not df.empty:
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
        return df

    def _generate_and_scale_target_for_a2(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if out.empty or 'kline_close' not in out.columns:
            self.logger.warning("A2 目标生成：输入df为空或无kline_close。")
            for col in ['dprice', config.A1_TARGET_COL_RAW_DPRICE_FOR_A2, config.A1_TARGET_COL_SCALED_DPRICE_FOR_A2]:
                out[col] = np.nan
            return out

        price_now = out['kline_close'].astype(float)
        out['dprice'] = price_now.diff().fillna(0.0)  # 当前区间的价格变化，作为特征

        # 预测目标：未来N个周期的价格变化
        price_fut = price_now.shift(-config.A1_HORIZON_PERIODS_FOR_A2)
        out[config.A1_TARGET_COL_RAW_DPRICE_FOR_A2] = price_fut - price_now  # 未缩放的目标

        if self.target_scaler_a2 is None:
            if self.scaler_a2_target_path.exists():  # 尝试再次加载
                try:
                    self.target_scaler_a2 = joblib.load(self.scaler_a2_target_path)
                    self.logger.info(f"A2 目标缩放器从 {self.scaler_a2_target_path} 重新加载成功。")
                except Exception as e_load:
                    self.logger.warning(f"重新加载 A2 目标缩放器失败: {e_load}。将尝试在线拟合。")

            if self.target_scaler_a2 is None:  # 如果还是None，则在线拟合
                self.target_scaler_a2 = StandardScaler(with_mean=False)
                fit_vals = out[config.A1_TARGET_COL_RAW_DPRICE_FOR_A2].dropna().values.reshape(-1, 1)
                if len(fit_vals) > 0:
                    self.target_scaler_a2.fit(fit_vals)
                    self.logger.info("A2 目标缩放器已在线拟合。")
                    if not self.scaler_a2_target_path.exists():  # 仅当文件不存在时保存
                        try:
                            self.scaler_a2_target_path.parent.mkdir(parents=True, exist_ok=True)
                            joblib.dump(self.target_scaler_a2, self.scaler_a2_target_path)
                            self.logger.info(f"A2 目标缩放器已保存到 {self.scaler_a2_target_path}")
                        except Exception as e_save:
                            self.logger.warning(f"保存 A2 目标缩放器失败: {e_save}")
                else:
                    self.logger.warning("A2 目标缩放器：无有效数据进行拟合，将保持未拟合状态。")
                    self.target_scaler_a2 = None

        if self.target_scaler_a2 and hasattr(self.target_scaler_a2,
                                             'scale_') and self.target_scaler_a2.scale_ is not None:
            mask_notna_target = out[config.A1_TARGET_COL_RAW_DPRICE_FOR_A2].notna()
            if mask_notna_target.any():
                transform_input = out.loc[mask_notna_target, config.A1_TARGET_COL_RAW_DPRICE_FOR_A2].values.reshape(-1,
                                                                                                                    1)
                out.loc[mask_notna_target, config.A1_TARGET_COL_SCALED_DPRICE_FOR_A2] = \
                    self.target_scaler_a2.transform(transform_input).flatten()
            else:
                out[config.A1_TARGET_COL_SCALED_DPRICE_FOR_A2] = np.nan
        else:
            out[config.A1_TARGET_COL_SCALED_DPRICE_FOR_A2] = np.nan
            if self.target_scaler_a2:
                self.logger.warning("A2 目标缩放器存在但未拟合，无法进行 transform。")
        return out

    def get_latest_processed_data_for_a201(self) -> Optional[Tuple[pd.DataFrame, Any]]:
        # 1) 保证全量链上缓存最新
        self.update_chain_state()  # 确保 self.chain_full_cache 是最新的

        # 2) 从内存缓存取最近 buffer_minutes 的行情 & 使用全量链上特征的副本
        # 为 a101.py 增加额外的 lookback 以处理初始 NaN
        a101_lookback_buffer_str = getattr(config, 'A101_PROCESSOR_ROLLING_BUFFER_ESTIMATE_STR', "0m")  # 从config获取
        try:
            a101_lookback_buffer_td = pd.Timedelta(a101_lookback_buffer_str)
        except ValueError:
            self.logger.warning(
                f"配置的 A101_PROCESSOR_ROLLING_BUFFER_ESTIMATE_STR ('{a101_lookback_buffer_str}') 无效，lookback buffer 设为0。")
            a101_lookback_buffer_td = pd.Timedelta(minutes=0)

        df_raw_m_full = self.raw_cache.get_market_data_copy()

        current_market_buffer_start_ts = self.market_buffer.index.min() if not self.market_buffer.empty else pd.Timestamp.now(
            tz=timezone.utc) - pd.Timedelta(minutes=self.buffer_minutes)
        effective_df_m_start_ts_a101 = current_market_buffer_start_ts - a101_lookback_buffer_td

        df_m_for_a101 = df_raw_m_full[df_raw_m_full.index >= effective_df_m_start_ts_a101].copy()
        df_c_for_a101 = self.chain_full_cache.copy()  # a101 通常需要更长的链上历史

        if df_m_for_a101.empty:  # df_c_for_a101可以为空，a101.resample_and_merge会处理
            self.logger.warning("A201 增量处理：应用lookback后，行情数据 (df_m_for_a101) 为空。")
            return None, None

        # 确保 df_m_for_a101 有 'volume' 列 (a101.resample_and_merge 需要)
        if 'volume' not in df_m_for_a101.columns and 'kline_volume' in df_m_for_a101.columns:
            df_m_for_a101['volume'] = df_m_for_a101['kline_volume']
        elif 'volume' not in df_m_for_a101.columns:
            self.logger.warning("A201 增量处理: df_m_for_a101 中缺少 'volume' 或 'kline_volume' 列。")
            df_m_for_a101['volume'] = np.nan  # 补一个NaN列，下游a101可能会报错或处理

        # 3) 调用 a101.py 的 compute_features (它内部会调用 resample_and_merge, generate_log_return_target, add_features, normalize_features)
        # compute_features 应该返回一个只包含模型输入特征的DataFrame
        try:
            # 注意：a101.compute_features 签名可能与此不符，需要适配
            # 这里假设 a101.compute_features 能处理原始的 df_m_for_a101 和 df_c_for_a101
            # 并且返回标准化后的、只含模型输入特征的DataFrame，以及scaler对象
            # 根据 a101.py 的 compute_features，它不直接返回 scaler，scaler是在 normalize_features 内部加载/使用的
            # 并且它不直接接受原始行情和链上数据，而是期望一个已合并的 DataFrame
            # 我们需要先调用 a101.resample_and_merge, a101.generate_log_return_target, a101.add_features, a101.normalize_features

            merged_a101 = resample_and_merge(df_m_for_a101, df_c_for_a101, agg_period=config.A101_AGG_PERIOD)
            if merged_a101.empty: self.logger.warning(
                "A201 增量处理：a101.resample_and_merge 结果为空。"); return None, None

            labeled_a101 = generate_log_return_target(merged_a101, pred_horizon_k=config.A101_PRED_HORIZON_K)
            if labeled_a101.empty: self.logger.warning(
                "A201 增量处理：a101.generate_log_return_target 结果为空。"); return None, None

            chain_cols_for_a101 = df_c_for_a101.columns.tolist() if not df_c_for_a101.empty else []
            featured_a101 = add_features(labeled_a101, chain_cols_original_names=chain_cols_for_a101,
                                         agg_period=config.A101_AGG_PERIOD,
                                         current_roll_window=config.A101_ROLL_WINDOW, is_training=False)
            if featured_a101.empty: self.logger.warning("A201 增量处理：a101.add_features 结果为空。"); return None, None

            normed_a101, scaler_a101_used = normalize_features(
                featured_a101,
                scaler_save_path=config.ARTIFACTS_DIR / config.A101_SCALER_FILENAME
            )
            if normed_a101.empty: self.logger.warning(
                "A201 增量处理：a101.normalize_features 结果为空。"); return None, None

            # normed_a101 应该包含了模型输入特征，以及 'close_raw', 'atr_raw', A101_TARGET_COL_NAME
            # 我们需要确保只把模型输入特征传递给模型
            # ModelPredictionService.features_a201 存储了模型期望的输入特征名
            model_input_features_a201 = self.features_a201
            if not model_input_features_a201:
                self.logger.error("A201 增量处理: self.features_a201 未设置。")
                return None, None

            # 确保 normed_a101 包含所有模型期望的输入特征
            missing_model_inputs = [f for f in model_input_features_a201 if f not in normed_a101.columns]
            if missing_model_inputs:
                self.logger.error(
                    f"A201 增量处理: a101处理流程后，normed_a101 缺失以下模型输入特征: {missing_model_inputs}。")
                # 可以尝试补0，但通常表示a101的scaler与模型checkpoint不匹配
                for col_miss in missing_model_inputs: normed_a101[col_miss] = 0.0

            # 截取模型需要的特征列，并确保顺序正确
            final_model_input_df_a201 = normed_a101[model_input_features_a201].copy()
            # 再次填充以防万一 (虽然 normalize_features 应该已经处理了)
            final_model_input_df_a201 = final_model_input_df_a201.ffill().bfill().fillna(0.0)


        except RuntimeError as e_a101_runtime:  # a101.normalize_features 可能会因scaler不匹配抛出 RuntimeError
            self.logger.error(f"A201 增量处理：a101.py 管道发生运行时错误 (可能scaler不匹配): {e_a101_runtime}",
                              exc_info=True)
            return None, None
        except Exception as e_a101:
            self.logger.error(f"A201 增量处理：a101.py 管道失败: {e_a101}", exc_info=True)
            return None, None

        # 7) 取最后 N 行构建滑窗并返回
        if len(final_model_input_df_a201) < self.seq_len_a201:
            self.logger.info(
                f"A201 增量处理：数据行数 ({len(final_model_input_df_a201)}) 少于窗口长度 ({self.seq_len_a201})。等待更多数据。")
            return None, None  # 返回两个 None

        self.logger.info(f"A201 增量处理：成功准备 {self.seq_len_a201} 行数据用于预测。")
        return final_model_input_df_a201.tail(
            self.seq_len_a201), scaler_a101_used  # scaler_a101_used 是从 normalize_features 返回的

    # —— 给 OnlineLearningManager 一次性生成 A2 训练块 ——
    def gen_a2_training_block(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp):
        market = self._load_market_from_db(start_ts, end_ts)
        chain = self._load_chain_from_db(start_ts, end_ts)

        # a1 级预处理
        df_a1 = self.processor_a1.get_processed_data_from_df(market, chain)
        if df_a1.empty:
            self.logger.warning("gen_a2_training_block：A1 处理结果为空")
            return None, None

        # 生成 ΔPrice 目标并做缩放
        df_a2 = self._generate_and_scale_target_for_a2(df_a1)

        # 保留模型需要的特征列 + 目标列
        if self.features_a2:
            need_cols = set(self.features_a2) | {
                config.A1_TARGET_COL_RAW_DPRICE_FOR_A2,
                config.A1_TARGET_COL_SCALED_DPRICE_FOR_A2,
            }
            df_a2 = df_a2[sorted(c for c in need_cols if c in df_a2.columns)]

        return df_a2, None  # scaler 已在 _generate_and_scale_target_for_a2 内部处理

    # —— 给 OnlineLearningManager 一次性生成 A201 训练块 ——
    def gen_a201_training_block(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp):
        market = self._load_market_from_db(start_ts, end_ts)
        chain = self._load_chain_from_db(start_ts, end_ts)

        merged = resample_and_merge(market, chain, agg_period=config.A101_AGG_PERIOD)
        labeled = generate_log_return_target(merged, pred_horizon_k=config.A101_PRED_HORIZON_K)
        featured = add_features(
            labeled,
            chain_cols_original_names=chain.columns.tolist(),
            agg_period=config.A101_AGG_PERIOD,
            current_roll_window=config.A101_ROLL_WINDOW,
            is_training=False,
        )
        normed, scaler_used = normalize_features(
            featured,
            scaler_save_path=config.ARTIFACTS_DIR / config.A101_SCALER_FILENAME,
        )
        if self.features_a201:
            normed = normed[self.features_a201 + ["target_logret", "target_dprice"]]

        return normed, scaler_used

    def _calc_start_dt_for_a201(self) -> str:
        # (保持不变)
        now = pd.Timestamp.utcnow()
        start_dt = now - pd.Timedelta(days=config.HISTORY_LOOKBACK_DAYS)
        return start_dt.strftime("%Y-%m-%d %H:%M:%S")

    def _calc_end_dt_for_a201(self) -> str:
        # (保持不变)
        now_utc = pd.Timestamp.utcnow()
        aligned_dt = now_utc.floor(f"{self.freq_minutes_a201}min")
        end_dt = aligned_dt - pd.Timedelta(minutes=self.freq_minutes_a201)
        return end_dt.strftime("%Y-%m-%d %H:%M:%S")

    # ───────── 兼容旧接口：供 OnlineLearningManager 调用 ─────────
    @property
    def a2_feature_list(self):
        """
        兼容早期代码使用 self.data_pipeline.a2_feature_list 的写法。
        实际仍然返回 features_a2。
        """
        return self.features_a2

    @a2_feature_list.setter
    def a2_feature_list(self, v):
        self.features_a2 = v

    @property
    def a201_feature_list(self):
        return self.features_a201

    @a201_feature_list.setter
    def a201_feature_list(self, v):
        self.features_a201 = v

class ModelPredictionService:
    def __init__(self, data_pipeline_manager_ref: DataPipelineManager):
        self.logger = logging.getLogger(f"{config.APP_NAME}.ModelPredictionService")
        self.device = config.DEVICE
        self.data_pipeline_manager = data_pipeline_manager_ref

        self.model_a2: Optional[ModelA2] = None
        self.features_a2: Optional[List[str]] = None
        self.input_dim_a2: Optional[int] = None
        self.a2_loaded_model_config: Optional[Dict[str, Any]] = None

        self.model_a201: Optional[ModelA201] = None
        self.features_a201: Optional[List[str]] = None
        self.input_dim_a201: Optional[int] = None
        self.a201_loaded_model_config: Optional[Dict[str, Any]] = None

        self._load_models_and_features()

        if self.features_a2 is not None:
            self.data_pipeline_manager.features_a2 = self.features_a2
            self.logger.info(f"已将 A2 的特征列表 (共{len(self.features_a2)}个) 设置到 DataPipelineManager。")
        else:
            self.logger.warning("A2 的特征列表 (self.features_a2) 未能成功加载，无法设置到 DataPipelineManager。")

        if self.features_a201 is not None:
            self.data_pipeline_manager.features_a201 = self.features_a201
            self.logger.info(f"已将 A201 的特征列表 (共{len(self.features_a201)}个) 设置到 DataPipelineManager。")
        else:
            self.logger.warning("A201 的特征列表 (self.features_a201) 未能成功加载，无法设置到 DataPipelineManager。")

        self.logger.info("ModelPredictionService 模型预测服务已初始化。")

    def _extract_features_from_df(
            self,
            df: pd.DataFrame,
            model_name: str,
            target_cols_to_exclude: List[str]
    ) -> List[str]:
        # (保持不变)
        if df is None or df.empty:
            return []
        base_exclude = (
                [n for n in df.index.names if n and n not in df.columns] +
                ['time_window', 'time_idx', 'timestamp'] +
                target_cols_to_exclude
        )
        if model_name == "a201":
            base_exclude.extend(['close_raw', 'atr_raw'] + config.A101_EXCLUDE_COLS)
        elif model_name == "a2":
            base_exclude.extend([
                config.A1_TARGET_COL_RAW_DPRICE_FOR_A2,
                config.A1_TARGET_COL_SCALED_DPRICE_FOR_A2,
                # 'dprice_scaled', # dprice (unscaled diff) can be a feature
                'mid_price_raw', 'micro_price_raw', 'spread_raw'
            ])
        final_exclude_set = set(base_exclude)
        feats_ok = [col for col in df.columns if col not in final_exclude_set]
        return feats_ok

    def reload_weights(self, model_name: str, weight_path: Path) -> None:
        # (保持不变)
        if model_name == "a2":
            model = self.model_a2
        elif model_name == "a201":
            model = self.model_a201
        else:
            self.logger.error(f"未知模型 {model_name}，无法热更新。"); return
        if model is None: self.logger.warning(f"{model_name} 未初始化，跳过热更新。"); return
        try:
            state = torch.load(weight_path, map_location=config.DEVICE)
            model.load_state_dict(state, strict=False);
            model.eval()
            self.logger.info(f"{model_name} 已热更新 -> {weight_path.name}")
        except Exception as exc:
            self.logger.exception(f"热更新 {model_name} 失败: {exc}")

    def _ensure_a2_model_built_if_needed(self, sample_df_for_a2: Optional[pd.DataFrame] = None):
        # (保持上一轮修复建议中的版本，特别是 input_dim 处理)
        if self.model_a2 is not None: return
        # —— 先从 .meta 文件读取训练截止时间 ——
        meta_path = Path(config.A2_MODEL_CHECKPOINT_PATH).with_suffix('.pth.meta')
        if meta_path.exists():
            try:
                raw = json.load(open(meta_path, "r"))
                config.TRAIN_END_TS_A2 = pd.to_datetime(raw["train_end_timestamp"], utc=True)
                self.logger.info(f"A2：从 .meta 加载训练截止时间 {config.TRAIN_END_TS_A2}")
            except Exception as e_meta:
                self.logger.warning(f"A2：读取 .meta 失败: {e_meta}")

        # —— 如果 .meta 不存在，则从 checkpoint 本体写入一次 ——
        if not meta_path.exists() and Path(config.A2_MODEL_CHECKPOINT_PATH).exists():
            try:
                ckpt0 = torch.load(config.A2_MODEL_CHECKPOINT_PATH, map_location="cpu")
                ts0 = ckpt0.get("train_end_timestamp")
                if ts0:
                    config.TRAIN_END_TS_A2 = pd.to_datetime(ts0, utc=True)
                    json.dump(
                        {"train_end_timestamp": str(config.TRAIN_END_TS_A2)},
                        open(meta_path, "w")
                    )
                    self.logger.info(f"A2：写入 .meta 训练截止时间 {config.TRAIN_END_TS_A2}")
            except Exception:
                pass


        if Path(config.A2_MODEL_CHECKPOINT_PATH).exists():
            self.logger.info(f"A2：尝试从 checkpoint {config.A2_MODEL_CHECKPOINT_PATH} 构建模型")
            try:
                ckpt = torch.load(config.A2_MODEL_CHECKPOINT_PATH, map_location='cpu')
                meta_path = Path(config.A2_MODEL_CHECKPOINT_PATH).with_suffix('.pth.meta')
                if meta_path.exists():
                    config.TRAIN_END_TS_A2 = pd.to_datetime(json.load(open(meta_path, "r"))["train_end_timestamp"],
                                                            utc=True)

                cfg_ckpt = ckpt.get('model_config')
                cols_ckpt = ckpt.get('input_cols')
                dim_ckpt = ckpt.get('input_dim')

                if cfg_ckpt and cols_ckpt and dim_ckpt is not None:
                    self.features_a2 = cols_ckpt
                    self.input_dim_a2 = dim_ckpt
                    # ---- 将训练截止时间写入全局 config ----
                    train_end_ts = ckpt.get("train_end_timestamp")
                    if train_end_ts:
                        #import config, pandas as pd
                        config.TRAIN_END_TS_A2 = pd.to_datetime(train_end_ts, utc=True)

                    cfg_use = cfg_ckpt.copy()
                    if 'input_dim' not in cfg_use or cfg_use.get('input_dim') != self.input_dim_a2:
                        if 'input_dim' in cfg_use and cfg_use.get('input_dim') is not None:
                            self.logger.warning(
                                f"A2：Checkpoint 的 model_config 中的 input_dim ({cfg_use.get('input_dim')}) "
                                f"与 checkpoint 直接记录的 input_dim ({self.input_dim_a2}) 不符。 "
                                f"将使用后者 ({self.input_dim_a2}) 并更新 model_config。"
                            )
                        elif 'input_dim' not in cfg_use:
                            self.logger.info(
                                f"A2: Checkpoint 的 model_config 中缺少 'input_dim'。将使用 checkpoint 直接记录的 input_dim ({self.input_dim_a2}) 更新 model_config。"
                            )
                        cfg_use['input_dim'] = self.input_dim_a2

                    cfg_use['use_gpu_wavelet'] = config.HAS_PYTORCH_WAVELETS and config.DEVICE.type == 'cuda'
                    self.a2_loaded_model_config = cfg_use.copy()
                    self.model_a2 = ModelA2(**cfg_use)

                    loaded_weights = False
                    if Path(config.A2_BEST_MODEL_PATH).exists():
                        try:
                            self.model_a2.load_state_dict(
                                torch.load(config.A2_BEST_MODEL_PATH, map_location=config.DEVICE))
                            loaded_weights = True;
                            self.logger.info(f"A2：已从 {config.A2_BEST_MODEL_PATH} 加载最优权重。")
                        except Exception as e:
                            self.logger.warning(
                                f"A2：加载最优权重 {config.A2_BEST_MODEL_PATH} 失败：{e}。尝试 checkpoint 内权重。")
                    if not loaded_weights:
                        if ckpt.get('ema_shadow_params'):
                            ema_decay_val = cfg_use.get('ema_decay', ckpt.get('training_args', {}).get('ema_decay',
                                                                                                       config.EMA_DECAY_DEFAULT))
                            ema = EMA(self.model_a2, decay=ema_decay_val)
                            ema.shadow_params = ckpt['ema_shadow_params']
                            ema.apply_shadow(self.model_a2)
                            self.logger.info(f"A2：已应用 checkpoint 中的 EMA 权重 (decay={ema_decay_val}).")
                        elif ckpt.get('model_state_dict'):
                            self.model_a2.load_state_dict(ckpt['model_state_dict'])
                            self.logger.info("A2：已从 checkpoint 加载 model_state_dict。")
                        else:
                            self.logger.warning(
                                "A2：Checkpoint 中未找到 'ema_shadow_params' 或 'model_state_dict'。模型权重未加载。")
                    if self.model_a2: self.model_a2.to(config.DEVICE).eval(); self.logger.info(
                        f"A2：模型已从 checkpoint 构建完成。输入维度={self.input_dim_a2}，特征数={len(self.features_a2)}。")
                    return
                else:
                    missing_keys_msg = [k for k, v_ in
                                        [('model_config', cfg_ckpt), ('input_cols', cols_ckpt), ('input_dim', dim_ckpt)]
                                        if v_ is None or (isinstance(v_, list) and not v_)]
                    self.logger.warning(
                        f"A2：Checkpoint {config.A2_MODEL_CHECKPOINT_PATH} 缺少关键信息: {missing_keys_msg}。")
            except Exception as e:
                self.logger.error(f"A2：从 checkpoint 构建模型失败: {e}", exc_info=True); self.model_a2 = None

        if self.model_a2 is None and sample_df_for_a2 is not None and not sample_df_for_a2.empty:
            self.logger.warning("A2：无法从 checkpoint 加载模型，尝试基于样本数据动态构建 (fallback)。")
            self.features_a2 = self._extract_features_from_df(sample_df_for_a2, "a2",
                                                              [config.A1_TARGET_COL_RAW_DPRICE_FOR_A2,
                                                               config.A1_TARGET_COL_SCALED_DPRICE_FOR_A2])
            if not self.features_a2: self.logger.error("A2：动态特征提取失败，无法构建模型。"); return
            self.input_dim_a2 = len(self.features_a2)
            params = config.A2_MODEL_CONSTRUCTOR_PARAMS_DEFAULT.copy()
            params['input_dim'] = self.input_dim_a2
            params['use_gpu_wavelet'] = config.HAS_PYTORCH_WAVELETS and config.DEVICE.type == 'cuda'
            self.a2_loaded_model_config = params.copy()
            try:
                self.model_a2 = ModelA2(**params)
                if Path(config.A2_BEST_MODEL_PATH).exists():
                    try:
                        self.model_a2.load_state_dict(
                            torch.load(config.A2_BEST_MODEL_PATH, map_location=config.DEVICE)); self.logger.info(
                            f"A2 (动态构建)：已从 {config.A2_BEST_MODEL_PATH} 加载最优权重。")
                    except Exception as e:
                        self.logger.error(
                            f"A2 (动态构建)：加载最优权重 {config.A2_BEST_MODEL_PATH} 失败：{e}。模型将使用初始化权重。")
                else:
                    self.logger.warning(
                        f"A2 (动态构建)：未找到最优权重文件 {config.A2_BEST_MODEL_PATH}。模型将使用初始化权重。")
                self.model_a2.to(config.DEVICE).eval();
                self.logger.info(f"A2：模型已动态构建完成。输入维度={self.input_dim_a2}，特征数={len(self.features_a2)}。")
            except Exception as e:
                self.logger.error(f"A2：动态构建模型失败: {e}",
                                  exc_info=True); self.model_a2 = None; self.features_a2 = None; self.input_dim_a2 = None
        if self.model_a2 is None: self.logger.error("A2：模型构建彻底失败（checkpoint 和动态构建均未成功）。")

    def _load_models_and_features(self):
        # (保持不变，它会调用 _ensure_a2_model_built_if_needed)
        self._ensure_a2_model_built_if_needed(sample_df_for_a2=None)
        # —— 先从 .meta 文件读取 A201 的训练截止时间 ——
        meta_a201 = Path(config.A201_MODEL_CHECKPOINT_PATH).with_suffix('.pth.meta')
        if meta_a201.exists():
            try:
                raw = json.load(open(meta_a201, "r"))
                config.TRAIN_END_TS_A201 = pd.to_datetime(raw["train_end_timestamp"], utc=True)
                self.logger.info(f"A201：从 .meta 加载训练截止时间 {config.TRAIN_END_TS_A201}")
            except Exception as e_meta:
                self.logger.warning(f"A201：读取 .meta 失败: {e_meta}")
        if Path(config.A201_MODEL_CHECKPOINT_PATH).exists():
            self.logger.info(f"A201：尝试从 checkpoint {config.A201_MODEL_CHECKPOINT_PATH} 构建模型。")
            try:
                ckpt = torch.load(config.A201_MODEL_CHECKPOINT_PATH, map_location='cpu')
                raw_ts = ckpt.get("train_end_timestamp")
                if raw_ts is None:
                    raw_ts = ckpt.get("processed_data_actual_end_time")
                if raw_ts:
                    config.TRAIN_END_TS_A201 = pd.to_datetime(raw_ts, utc=True)
                    # 同步写入 .pth.meta 以便下次直接读取
                    meta_path = Path(config.A201_MODEL_CHECKPOINT_PATH).with_suffix('.pth.meta')
                    with open(meta_path, "w") as mf:
                        json.dump({"train_end_timestamp": str(config.TRAIN_END_TS_A201)}, mf)

                cfg_ckpt = ckpt.get('model_config')
                feats_ckpt = ckpt.get('selected_features_names')
                dim_ckpt = ckpt.get('feat_dim') or ckpt.get('input_dim')

                if cfg_ckpt and feats_ckpt and dim_ckpt is not None:
                    self.features_a201 = feats_ckpt
                    self.input_dim_a201 = dim_ckpt
                    train_end_ts = ckpt.get("train_end_timestamp")
                    if train_end_ts:
                       # import config, pandas as pd
                        config.TRAIN_END_TS_A201 = pd.to_datetime(train_end_ts, utc=True)

                    cfg_use = cfg_ckpt.copy()
                    if cfg_use.get('in_dim') != self.input_dim_a201:
                        self.logger.warning(
                            f"A201: Checkpoint model_config 'in_dim' ({cfg_use.get('in_dim')}) 与 feat_dim ({self.input_dim_a201}) 不符，使用后者。")
                        cfg_use['in_dim'] = self.input_dim_a201
                    self.a201_loaded_model_config = cfg_use.copy()
                    self.model_a201 = ModelA201(**cfg_use)
                    loaded_weights = False
                    if Path(config.A201_BEST_MODEL_PATH).exists():
                        try:
                            self.model_a201.load_state_dict(
                                torch.load(config.A201_BEST_MODEL_PATH, map_location=config.DEVICE))
                            loaded_weights = True;
                            self.logger.info(f"A201：已从 {config.A201_BEST_MODEL_PATH} 加载最优权重。")
                        except Exception as e:
                            self.logger.warning(
                                f"A201：加载最优权重 {config.A201_BEST_MODEL_PATH} 失败：{e}。尝试 checkpoint 内权重。")
                    if not loaded_weights:
                        if ckpt.get('ema_shadow_params'):
                            ema_decay = ckpt.get('training_args', {}).get('ema_decay', config.EMA_DECAY_DEFAULT)
                            ema = EMA(self.model_a201, decay=ema_decay);
                            ema.load_shadow(ckpt['ema_shadow_params']);
                            ema.apply_shadow(self.model_a201)
                            self.logger.info(f"A201：已应用 checkpoint 中的 EMA 权重 (decay={ema_decay}).")
                        elif ckpt.get('model_state_dict'):
                            self.model_a201.load_state_dict(ckpt['model_state_dict']);
                            self.logger.info("A201：已从 checkpoint 加载 model_state_dict。")
                        else:
                            self.logger.warning(
                                "A201：Checkpoint 中未找到 'ema_shadow_params' 或 'model_state_dict'。模型权重未加载。"); self.model_a201 = None
                    if self.model_a201: self.model_a201.to(config.DEVICE).eval(); self.logger.info(
                        f"A201：模型已从 checkpoint 构建完成。输入维度={self.input_dim_a201}，特征数={len(self.features_a201)}。")
                else:
                    missing_keys_msg = [k for k, v_ in
                                        [('model_config', cfg_ckpt), ('selected_features_names', feats_ckpt),
                                         ('feat_dim/input_dim', dim_ckpt)] if
                                        v_ is None or (isinstance(v_, list) and not v_)]
                    self.logger.error(
                        f"A201：Checkpoint {config.A201_MODEL_CHECKPOINT_PATH} 缺少关键信息: {missing_keys_msg}。模型构建失败。");
                    self.model_a201 = None
            except Exception as e:
                self.logger.error(f"A201：从 checkpoint 构建模型失败: {e}", exc_info=True); self.model_a201 = None
        else:
            self.logger.warning(
                f"A201：未找到 checkpoint 文件 {config.A201_MODEL_CHECKPOINT_PATH}。A201 模型将不可用。"); self.model_a201 = None

    def _prepare_input_tensor(
            self, df: pd.DataFrame, feats: List[str],
            seq_len: int, model_name_log: str
    ) -> Optional[torch.Tensor]:
        # (保持不变)
        if df is None: self.logger.warning(f"{model_name_log}：输入 DataFrame 为 None，无法准备 Tensor。"); return None
        if len(df) < seq_len: self.logger.warning(
            f"{model_name_log}：数据行数 ({len(df)}) 不足序列长度 ({seq_len})，无法准备 Tensor。"); return None
        if not feats: self.logger.error(f"{model_name_log}：特征列表为空，无法准备 Tensor。"); return None
        missing_feats = [f for f in feats if f not in df.columns]
        if missing_feats:
            self.logger.error(
                f"{model_name_log}: 输入数据缺少以下特征 ({len(missing_feats)}): {missing_feats[:8]}{'...' if len(missing_feats) > 8 else ''}. 无法准备 Tensor.")
            return None
        df_selected_feats = df[feats].tail(seq_len).copy()
        if df_selected_feats.isnull().values.any():
            nan_cols_summary = df_selected_feats.isnull().sum()
            nan_cols = nan_cols_summary[nan_cols_summary > 0].index.tolist()
            self.logger.warning(f"{model_name_log}: 输入序列在以下列中包含 NaN 值: {nan_cols}. 将尝试使用列中位数填充。")
            for col in nan_cols:
                median_val = df_selected_feats[col].median()
                if pd.notna(median_val):
                    df_selected_feats[col].fillna(median_val, inplace=True); self.logger.debug(
                        f"{model_name_log}: 列 '{col}' 中的 NaN 已用中位数 {median_val:.4f} 填充。")
                else:
                    df_selected_feats[col].fillna(0.0, inplace=True); self.logger.warning(
                        f"{model_name_log}: 列 '{col}' 中位数不可用，NaN 已用 0.0 填充。")
            if df_selected_feats.isnull().values.any(): self.logger.error(
                f"{model_name_log}: NaN填充后仍存在NaN值，Tensor准备失败。"); return None
        try:
            tensor_data = df_selected_feats.values.astype(np.float32)
            input_tensor = torch.tensor(tensor_data, dtype=torch.float32, device=self.device).unsqueeze(0)
            return input_tensor
        except Exception as e:
            self.logger.error(f"{model_name_log}: 从 DataFrame 创建 Tensor 失败: {e}", exc_info=True); return None

    def predict_a2(self, df_proc_a1: pd.DataFrame) -> Optional[float]:
        # (保持不变)
        if self.model_a2 is None: self._ensure_a2_model_built_if_needed(df_proc_a1)
        if self.model_a2 is None: self.logger.error("A2 模型未加载或构建失败，无法进行预测。"); return None
        if not self.features_a2: self.logger.error("A2 模型特征列表未设置，无法进行预测。"); return None
        tensor_in = self._prepare_input_tensor(df_proc_a1, self.features_a2, config.A2_SEQ_LEN, "A2")
        if tensor_in is None: self.logger.error("A2 输入 Tensor 准备失败，取消预测。"); return None
        try:
            with torch.no_grad():
                pred_scaled = self.model_a2(tensor_in, None)
            scaler = self.data_pipeline_manager.target_scaler_a2
            if scaler and hasattr(scaler, 'scale_') and scaler.scale_ is not None:
                try:
                    pred_unscaled_np = scaler.inverse_transform(pred_scaled.cpu().numpy().reshape(-1, 1))
                    return float(pred_unscaled_np.item())
                except Exception as e:
                    self.logger.error(f"A2 预测结果反向缩放失败: {e}. 返回原始缩放值。"); return pred_scaled.item()
            else:
                self.logger.warning("A2 目标缩放器不可用或未拟合，返回原始缩放预测值。"); return pred_scaled.item()
        except Exception as e:
            self.logger.error(f"A2 模型预测过程中发生错误: {e}", exc_info=True); return None

    def predict_a201(self, df_proc_a101: pd.DataFrame) -> Optional[Tuple[float, float]]:
        # (保持不变)
        if self.model_a201 is None: self.logger.error("A201 模型未加载，无法进行预测。"); return None
        if not self.features_a201: self.logger.error("A201 模型特征列表未设置，无法进行预测。"); return None
        tensor_in = self._prepare_input_tensor(df_proc_a101, self.features_a201, config.A201_SEQ_LEN, "A201")
        if tensor_in is None: self.logger.error("A201 输入 Tensor 准备失败，取消预测。"); return None
        try:
            with torch.no_grad():
                lr_t, dp_enc_t = self.model_a201(tensor_in)
            dp_dec_val = dprice_decode_a201(dp_enc_t)
            lr_val = lr_t.item()
            dp_dec_item = dp_dec_val.item() if isinstance(dp_dec_val, torch.Tensor) else float(dp_dec_val)
            return lr_val, dp_dec_item
        except Exception as e:
            self.logger.error(f"A201 模型预测或解码过程中发生错误: {e}", exc_info=True); return None