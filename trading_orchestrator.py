# trading_orchestrator.py
import logging
import sys
import threading
from datetime import datetime
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import hashlib  # Keep for MasterController's feature hashing
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pytz
import torch
from sqlalchemy import create_engine, Engine

# 从拆分出的模块导入
import config  # Import the whole config module
from core_utils import RollingRawCache, SignalGenerator
from data_services import DataPipelineManager, ModelPredictionService
from learning_services import OnlineLearningManager, ErrorSweeper
#from datetime import timedelta
# 外部模块的检查导入 (主要用于启动时的日志)
try:
    from config import HAS_PYTORCH_WAVELETS  # Already in config
except:
    HAS_PYTORCH_WAVELETS = False  # Fallback
logging.basicConfig(
    level=config.LOG_LEVEL,
    format=config.LOG_FORMAT_CN
)

# Path objects for checks
A2_BEST_MODEL_PATH_CHECK = config.ARTIFACTS_DIR / "best_model.pth"  # Example, use actual config path
A2_MODEL_CHECKPOINT_PATH_CHECK = config.ARTIFACTS_DIR / "checkpoint_best.pth"
A201_MODEL_CHECKPOINT_PATH_CHECK = config.ARTIFACTS_DIR / "best_ckpt_pytorch_lgbm60_mt_logret_dprice_K1_agg3min_rw15_bktmix.pth"


class MasterController:
    def __init__(self):
        self.logger = logging.getLogger(f"{config.APP_NAME}.MasterController")

        self.db_engine: Engine = create_engine(
            config.DATABASE_URL,
            pool_size=config.DB_ENGINE_POOL_SIZE,
            max_overflow=config.DB_ENGINE_MAX_OVERFLOW,
            pool_recycle=3600
        )
        self.logger.info(f"全局数据库引擎已创建。连接池大小：{config.DB_ENGINE_POOL_SIZE}，最大溢出：{config.DB_ENGINE_MAX_OVERFLOW}")


        self.raw_cache = RollingRawCache(max_hours=config.RAW_CACHE_MAX_HOURS)
        self.data_pipeline = DataPipelineManager(raw_cache_ref=self.raw_cache, db_engine=self.db_engine)
        self.model_service = ModelPredictionService(data_pipeline_manager_ref=self.data_pipeline)
        self.data_pipeline.features_a2 = self.model_service.features_a2
        self.data_pipeline.features_a201 = self.model_service.features_a201
        self.signal_generator = SignalGenerator()
        self.online_learner = OnlineLearningManager(model_service_ref=self.model_service)
        # 启动在线学习后台线程（非阻塞预测）
        threading.Thread(
            target=self.online_learner.training_daemon,
            name="OnlineTrainer",
            daemon=True
        ).start()

        self.error_sweeper = ErrorSweeper(
            online_learner_ref=self.online_learner,
            data_pipeline_ref=self.data_pipeline,
            model_service_ref=self.model_service,
            db_engine=self.db_engine
        )

        self.seq_len_a2 = config.A2_SEQ_LEN
        self.seq_len_a201 = config.A201_SEQ_LEN
        self.last_processed_index_a2: Optional[pd.Timestamp] = None
        self.last_prediction_a2: Optional[float] = None
        self.last_features_hash_a2: Optional[str] = None
        self.last_processed_index_a201: Optional[pd.Timestamp] = None
        self.last_prediction_a201: Optional[Tuple[float, float]] = None
        self.last_features_hash_a201: Optional[str] = None

        self.logger.info("MasterController 初始化完成.")
        if config.A1_FREQ != "30s": self.logger.warning(f"A1_FREQ({config.A1_FREQ}) non-standard.")
        if config.A101_AGG_PERIOD != "3min": self.logger.warning(
            f"A101_AGG_PERIOD({config.A101_AGG_PERIOD}) non-standard.")
    # ===== 预测包装与日志 =====

    def _hash_features(self, df_tail: pd.DataFrame) -> str:
        """对最近 seq_len 行特征做 md5，避免同批数据重复预测。"""
        return hashlib.md5(df_tail.values.tobytes()).hexdigest()

    def predict_and_log_a2(self, df_proc_a1: pd.DataFrame) -> None:
        slice_feat = df_proc_a1.tail(self.seq_len_a2)

        slice_feat_num = slice_feat.select_dtypes(include=[np.number]).copy()
        # 按训练期特征顺序重排，防止列错位
        slice_feat_num = slice_feat_num[self.model_service.features_a2]
        feat_hash = self._hash_features(slice_feat_num)
        if feat_hash == self.last_features_hash_a2:
            self.logger.debug("A2：特征未变化，跳过重复预测。")
            return
        self.last_features_hash_a2 = feat_hash

        pred_dp = self.model_service.predict_a2(slice_feat)
        if pred_dp is None:
            self.logger.warning("A2：预测失败。"); return

        ts_pred = slice_feat.index[-1]
        self.last_processed_index_a2 = ts_pred
        self.last_prediction_a2 = pred_dp

        # 1) 记录预测
        self.error_sweeper.record_prediction(
            model_name="a2",
            ts_pred=ts_pred,
            y_hat_logret=None,
            y_hat_dprice=pred_dp,
            raw_features_for_learning=slice_feat_num.values.astype("float32"),
        )
        # 2) 在线学习缓存
        x_tensor = torch.tensor(slice_feat_num.values, dtype=torch.float32)
        y_tensor = torch.tensor([pred_dp], dtype=torch.float32)
        self.online_learner.add_new_sample_for_learning("a2", x_tensor, y_tensor)

        # 3) 生成信号
        sig = self.signal_generator.generate_signal_from_a2(pred_dp)
       # self.signal_generator.collect_signal(sig)

    def predict_and_log_a201(self, df_proc_a101: pd.DataFrame, scaler_a201) -> None:
        slice_feat = df_proc_a101.tail(self.seq_len_a201)

        slice_feat_num = slice_feat.select_dtypes(include=[np.number]).copy()
        slice_feat_num = slice_feat_num[self.model_service.features_a201]
        feat_hash = self._hash_features(slice_feat_num)

        if feat_hash == self.last_features_hash_a201:
            self.logger.debug("A201：特征未变化，跳过重复预测。")
            return
        self.last_features_hash_a201 = feat_hash

        preds = self.model_service.predict_a201(slice_feat)
        if preds is None:
            self.logger.warning("A201：预测失败。"); return
        lr_pred, dp_pred = preds
        ts_pred = slice_feat.index[-1]
        self.last_processed_index_a201 = ts_pred
        self.last_prediction_a201 = preds

        # 1) 记录预测
        self.error_sweeper.record_prediction(
            model_name="a201",
            ts_pred=ts_pred,
            y_hat_logret=lr_pred,
            y_hat_dprice=dp_pred,
            raw_features_for_learning = slice_feat_num.values.astype("float32")
,
        )
        # 2) 在线学习缓存
        x_tensor = torch.tensor(slice_feat_num.values, dtype=torch.float32)
        y_lr = torch.tensor([lr_pred], dtype=torch.float32)
        y_dp = torch.tensor([dp_pred], dtype=torch.float32)
        self.online_learner.add_new_sample_for_learning("a201", x_tensor, y_lr, y_dp)

        # 3) 生成信号
        sig = self.signal_generator.generate_signal_from_a201(preds)
        #self.signal_generator.collect_signal(sig)

    def run_single_cycle(self) -> None:
        """
        单次主循环：
          1. 调度 A2 数据 → 预测 ΔPrice
          2. 调度 A201 数据 → 预测 3 min LogRet / ΔPrice
          3. 生成综合交易信号

        """
        cycle_start_utc = datetime.utcnow()
        self.logger.info("主循环开始： %s", cycle_start_utc)

        try:
            # ---------- A2 ----------
            df_a2 = self.data_pipeline.get_latest_processed_data_for_a2()
            if df_a2 is None:
                self.logger.warning("本轮缺少 A2 数据，跳过 A2 预测")
            else:
                self.predict_and_log_a2(df_a2)
            self.data_pipeline.update_caches()

            # ---------- A2 ----------
            df_a2 = self.data_pipeline.get_latest_processed_data_for_a2()
            # … A2 预测相关逻辑 …

            # ——— 在 A201 数据处理前，先增量更新链上缓存 ———
            self.data_pipeline.update_caches()

            # ---------- A201 ----------
            df_a201, scaler_a201 = self.data_pipeline.get_latest_processed_data_for_a201()

            # ---------- A201 ----------
            df_a201, scaler_a201 = self.data_pipeline.get_latest_processed_data_for_a201()
            if df_a201 is None:
                self.logger.warning("本轮缺少 A201 数据，跳过 A201 预测")
            else:
                self.predict_and_log_a201(df_a201, scaler_a201)

            # ---------- 生成综合信号 ----------
            # 用上一步预测结果，调用 combine_signals 生成最终信号
            sig_a2 = self.signal_generator.generate_signal_from_a2(self.last_prediction_a2)
            sig_a201 = self.signal_generator.generate_signal_from_a201(self.last_prediction_a201)
            final_sig = self.signal_generator.combine_signals(sig_a2, sig_a201)
            # 这里可以打印或下单
            self.logger.info(f"综合信号: {final_sig}")


        except Exception as exc:
            self.logger.error("主循环错误： %s", exc, exc_info=True)

        finally:
            cycle_end_utc = datetime.utcnow()
            elapsed = (cycle_end_utc - cycle_start_utc).total_seconds()
            self.logger.info("主循环结束： %s，本轮耗时 %.2f 秒", cycle_end_utc, elapsed)

    def start_main_loop(self):
        self.logger.info(f"主循环启动，周期： loop starting. Tick: {config.MAIN_TICK_INTERVAL_SECONDS}s.")
        self.error_sweeper.start()
        if config.DEVICE.type == "cuda" and config.USE_CUDNN_BENCHMARK:
            torch.backends.cudnn.benchmark = True;
            self.logger.info("已启用 CUDNN benchmark")
        else:
            torch.backends.cudnn.benchmark = False

        try:
            while True:
                loop_start = time.time()
                try:
                    self.run_single_cycle()
                except Exception as e:
                    self.logger.error(f"主循环错误： {e}", exc_info=True)
                elapsed = time.time() - loop_start
                sleep_for = config.MAIN_TICK_INTERVAL_SECONDS - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    self.logger.warning(f"本轮耗时 {elapsed:.2f}s, over budget!")
        except KeyboardInterrupt:
            self.logger.info("收到用户中断信号")
        except Exception as e:
            self.logger.critical(f"MasterController 致命错误： {e}", exc_info=True)
        finally:
            self.logger.info("MasterController 正在停止...")
            if self.error_sweeper.is_alive():
                self.error_sweeper.stop();
                self.error_sweeper.join(timeout=10)
            if self.db_engine: self.db_engine.dispose(); self.logger.info("DB engine disposed.")
            self.logger.info("MasterController stopped.")


if __name__ == "__main__":
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)  # Ensure artifacts dir exists
    logging.basicConfig(
        stream=sys.stdout, level=config.LOG_LEVEL,
        format=config.LOG_FORMAT_CN,
        datefmt='%Y-%m-%d %H:%M:%S', force=True
    )
    logger_main = logging.getLogger(f"{config.APP_NAME}.MainEntry")
    logger_main.info(f"应用『{config.APP_NAME}』启动")
    logger_main.info(f"检测到运行设备：{config.DEVICE_STR.upper()}")

    if not HAS_PYTORCH_WAVELETS and config.A2_MODEL_CONSTRUCTOR_PARAMS_DEFAULT.get('use_gpu_wavelet', False):
        logger_main.warning("请求使用 GPU-wavelet，但未安装 pytorch-wavelets")
    if not config.A2_BEST_MODEL_PATH.exists(): logger_main.warning(f"A2 weights {config.A2_BEST_MODEL_PATH} 缺失！")
    if not config.A2_MODEL_CHECKPOINT_PATH.exists(): logger_main.warning(
        f"A2 checkpoint {config.A2_MODEL_CHECKPOINT_PATH} 缺失！")
    if not config.A201_MODEL_CHECKPOINT_PATH.exists(): logger_main.warning(
        f"A201 checkpoint {config.A201_MODEL_CHECKPOINT_PATH} 缺失！")

    controller = MasterController()
    controller.start_main_loop()