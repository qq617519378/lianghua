# learning_services.py (修正版 - 禁用在线学习 - 中文日志/注释)
import json
import logging
import random
import threading
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Type
from collections import deque
import time
import os
import io
import copy
import functools
from datetime import timedelta

from sqlalchemy import text

from datetime import timezone  # 用于 timezone.utc

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sqlalchemy import text, inspect, Engine, MetaData, Table
from sqlalchemy.exc import SQLAlchemyError
from queue import Queue, Empty

# ────────────────────────────────────────────────
# 本地模块导入
# ────────────────────────────────────────────────
import config # 导入整个config模块，以便使用config.ENABLE_ONLINE_LEARNING


from core_utils import EMA

# 模型与损失 - weighted_loss_a2 的导入和备用逻辑
from a2 import HybridNeuralODEModel as ModelA2, weighted_loss as weighted_loss_a2, log_cosh as log_cosh_a2

from a201 import (
    PricePredictorModel as ModelA201,
    smooth_l1 as smooth_l1_a201,
    dprice_encode as dprice_encode_a201,
)

# 向后兼容保护：如果后面还有旧常量写死，直接抛异常提醒
assert 'MIN_SAMPLES_FOR_BATCH_LEARNING' not in globals(), \
    "请不要再使用 MIN_SAMPLES_FOR_BATCH_LEARNING——改用 ONLINE_BATCH_SIZE"

class ErrorSweeper(threading.Thread):
    def __init__(
            self,
            online_learner_ref,  # : OnlineLearningManager,
            data_pipeline_ref,  # : DataPipelineManager,
            model_service_ref,  # : ModelPredictionService,
            db_engine: Engine,
    ):
        super().__init__(daemon=True)
        self.logger = logging.getLogger(f"{config.APP_NAME}.ErrorSweeper")
        self.online_learner = online_learner_ref
        self.data_pipeline = data_pipeline_ref
        self.model_service = model_service_ref
        self.db_engine = db_engine
        self._stop_event = threading.Event()
        self._queue: Queue = Queue()  # 用于从 record_prediction 接收原始预测记录
        self.sweep_interval = config.ERROR_SWEEPER_INTERVAL_SECONDS

        self._ensure_prediction_log_table()
        self._meta = MetaData()

        try:
            self._log_tbl = Table(
                config.DB_PREDICTION_LOG_TABLE,
                self._meta,
                autoload_with=self.db_engine
            )
            self.logger.info(f"[错误清扫器] 表 '{config.DB_PREDICTION_LOG_TABLE}' 元数据加载完毕。")
        except Exception as e_table_load:
            self.logger.critical(
                f"[错误清扫器] 严重错误：初始化或反射表 '{config.DB_PREDICTION_LOG_TABLE}' 失败。错误清扫器无法工作：{e_table_load}",
                exc_info=True)
            raise RuntimeError(f"错误清扫器初始化表 {config.DB_PREDICTION_LOG_TABLE} 失败") from e_table_load

        self.a1_horizon_seconds_for_a2_target = int(
            pd.Timedelta(config.A1_FREQ).total_seconds() * config.A1_HORIZON_PERIODS_FOR_A2
        )
        self.a201_horizon_seconds = int(
            pd.Timedelta(config.A101_AGG_PERIOD).total_seconds() * config.A101_PRED_HORIZON_K
        )

        self.logger.info(f"错误清扫器 (ErrorSweeper) 已初始化。扫描间隔：{self.sweep_interval} 秒。")

    def _ensure_prediction_log_table(self):
        insp = inspect(self.db_engine)
        if not insp.has_table(config.DB_PREDICTION_LOG_TABLE):
            self.logger.info(f"表 '{config.DB_PREDICTION_LOG_TABLE}' 未找到，正在创建。")
            try:
                with self.db_engine.connect() as conn:
                    conn.execute(text(f"""
                        CREATE TABLE {config.DB_PREDICTION_LOG_TABLE} (
                            id SERIAL PRIMARY KEY,
                            model_name VARCHAR(50) NOT NULL,
                            ts_pred TIMESTAMP WITH TIME ZONE NOT NULL,
                            ts_target TIMESTAMP WITH TIME ZONE NOT NULL,
                            y_hat_logret FLOAT,
                            y_hat_dprice FLOAT,
                            y_true_logret FLOAT,
                            y_true_dprice FLOAT,
                            error_logret FLOAT,
                            error_dprice FLOAT,
                            evaluated BOOLEAN DEFAULT FALSE,
                            raw_features_pickle BYTEA,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                        );
                    """))
                    conn.execute(text(f"""
                        CREATE INDEX IF NOT EXISTS idx_pred_log_ts_target_eval
                        ON {config.DB_PREDICTION_LOG_TABLE}(ts_target, evaluated);
                    """))
                    conn.commit()
                self.logger.info(f"表 '{config.DB_PREDICTION_LOG_TABLE}' 已创建。")
            except Exception as e:
                self.logger.error(f"创建表 '{config.DB_PREDICTION_LOG_TABLE}' 失败: {e}", exc_info=True)
                raise

        cols_info = inspect(self.db_engine).get_columns(config.DB_PREDICTION_LOG_TABLE)
        existing = {col['name'] for col in cols_info}
        alter_stmts = []

        if 'y_hat_price' not in existing:
            alter_stmts.append(
                f"ALTER TABLE {config.DB_PREDICTION_LOG_TABLE} ADD COLUMN y_hat_price FLOAT"
            )
        if 'y_true_price' not in existing:
            alter_stmts.append(
                f"ALTER TABLE {config.DB_PREDICTION_LOG_TABLE} ADD COLUMN y_true_price FLOAT"
            )
        if 'error_price' not in existing:
            alter_stmts.append(
                f"ALTER TABLE {config.DB_PREDICTION_LOG_TABLE} ADD COLUMN error_price FLOAT"
            )

        if alter_stmts:
            with self.db_engine.begin() as conn:
                for stmt in alter_stmts:
                    self.logger.info(f"[错误清扫器] 执行表结构更新：{stmt}")
                    conn.execute(text(stmt))
            added = [stmt.rsplit(" ", 1)[-1] for stmt in alter_stmts]
            self.logger.info(
                f"[错误清扫器] 已在表 '{config.DB_PREDICTION_LOG_TABLE}' 中补齐列：{', '.join(added)}"
            )

    def record_prediction(self, model_name: str, ts_pred: pd.Timestamp,
                          y_hat_logret: Optional[float], y_hat_dprice: Optional[float],
                          raw_features_for_learning: Optional[np.ndarray] = None):
        horizon_sec = self.a1_horizon_seconds_for_a2_target if model_name == "a2" else self.a201_horizon_seconds
        ts_target = ts_pred + pd.Timedelta(seconds=horizon_sec)
        price_at_pred = self._get_true_price_at_timestamp(ts_pred, model_name)
        pred_price = (price_at_pred + y_hat_dprice) if price_at_pred is not None and y_hat_dprice is not None else None

        features_bytes = None
        if raw_features_for_learning is not None:
            try:
                with io.BytesIO() as bio:
                    np.savez_compressed(bio, features_array=raw_features_for_learning)
                    features_bytes = bio.getvalue()
            except Exception as e_compress:
                self.logger.error(f"特征压缩失败 (模型 {model_name} 时间 {ts_pred}): {e_compress}")

        record_to_queue = {
            "model_name": model_name,
            "ts_pred": ts_pred.to_pydatetime(),
            "ts_target": ts_target.to_pydatetime(),
            "y_hat_logret": y_hat_logret,
            "y_hat_dprice": y_hat_dprice,
            "y_hat_price": pred_price,
            "raw_features_pickle": features_bytes,
            "y_true_logret": None,
            "y_true_dprice": None,
            "error_logret": None,
            "error_dprice": None,
            "evaluated": False,
        }
        self._queue.put(record_to_queue)

    def run(self):
        self.logger.info("[错误清扫器] 线程已启动。")
        while not self._stop_event.is_set():
            records_batch_from_queue = []
            try:
                for _ in range(100):
                    if self._queue.empty(): break
                    records_batch_from_queue.append(self._queue.get_nowait())
            except Empty:
                pass

            if records_batch_from_queue:
                try:
                    with self.db_engine.begin() as conn:
                        if self._log_tbl is None:
                            self.logger.error("[错误清扫器] self._log_tbl 在运行时为 None，无法插入数据。")
                        else:
                            conn.execute(self._log_tbl.insert(), records_batch_from_queue)
                    for _ in range(len(records_batch_from_queue)): self._queue.task_done()
                    self.logger.debug(
                        f"[错误清扫器] 已向数据库插入 {len(records_batch_from_queue)} 条预测记录。")
                except SQLAlchemyError as e:
                    self.logger.error(f"[错误清扫器] 数据库批量插入失败：{e}。记录仍保留在队列中。",
                                      exc_info=True)
                except AttributeError as ae:
                    self.logger.critical(
                        f"[错误清扫器] 严重错误：self._log_tbl 未正确初始化，无法插入数据。错误：{ae}",
                        exc_info=True)
                except Exception as ex:
                    self.logger.error(f"[错误清扫器] 数据库插入期间发生意外错误：{ex}", exc_info=True)

            try:
                now_utc = pd.Timestamp.now(tz=timezone.utc)
                fetched_rows_for_eval = []
                with self.db_engine.connect() as conn_eval:
                    query_eval = text(
                        f"""
                        SELECT
                            id, model_name, ts_pred, ts_target,
                            y_hat_logret, y_hat_dprice, y_hat_price, raw_features_pickle
                        FROM {config.DB_PREDICTION_LOG_TABLE}
                        WHERE ts_target <= :now AND evaluated = FALSE
                        ORDER BY ts_target ASC
                        LIMIT 50
                        """
                    )
                    result_proxy = conn_eval.execute(query_eval, {"now": now_utc})
                    fetched_rows_for_eval = result_proxy.fetchall()

                if fetched_rows_for_eval:
                    self.logger.info(f"[错误清扫器] 发现 {len(fetched_rows_for_eval)} 条记录待错误评估。")
                    update_params_list_db = []
                    for r_eval in fetched_rows_for_eval:
                        ts_pred_pd = pd.Timestamp(r_eval.ts_pred)
                        ts_target_pd = pd.Timestamp(r_eval.ts_target)

                        if ts_pred_pd.tzinfo is None: ts_pred_pd = ts_pred_pd.tz_localize(timezone.utc)
                        elif ts_pred_pd.tzinfo != timezone.utc: ts_pred_pd = ts_pred_pd.tz_convert(timezone.utc)
                        if ts_target_pd.tzinfo is None: ts_target_pd = ts_target_pd.tz_localize(timezone.utc)
                        elif ts_target_pd.tzinfo != timezone.utc: ts_target_pd = ts_target_pd.tz_convert(timezone.utc)

                        price_at_pred_time = self._get_true_price_at_timestamp(ts_pred_pd, r_eval.model_name)
                        price_at_target_time = self._get_true_price_at_timestamp(ts_target_pd, r_eval.model_name)

                        if price_at_pred_time is None or price_at_target_time is None:
                            self.logger.warning(f"无法获取 ID {r_eval.id} 的真实价格，跳过评估。")
                            continue

                        yt_dp = price_at_target_time - price_at_pred_time
                        err_dp = (yt_dp - float(r_eval.y_hat_dprice)) if r_eval.y_hat_dprice is not None else None
                        ytp = price_at_target_time
                        err_price = (price_at_target_time - float(r_eval.y_hat_price)) if r_eval.y_hat_price is not None else None

                        yt_lr, err_lr = None, None
                        if abs(price_at_pred_time) > 1e-9:
                            yt_lr = np.log(price_at_target_time / price_at_pred_time) if abs(price_at_target_time) > 1e-9 else 0.0
                            if r_eval.y_hat_logret is not None: err_lr = yt_lr - float(r_eval.y_hat_logret)

                        update_params_list_db.append({
                            "ytlr": yt_lr, "ytdp": yt_dp, "ytp": ytp,
                            "elr": err_lr, "edp": err_dp, "error_price": err_price,
                            "id_val": r_eval.id, "evaluated_status": True
                        })

                        if r_eval.raw_features_pickle and self.online_learner is not None:
                            try:
                                with io.BytesIO(r_eval.raw_features_pickle) as bio:
                                    data_loaded = np.load(bio, allow_pickle=True)
                                    feats_arr = data_loaded['features_array']
                                feats_t = torch.tensor(feats_arr, dtype=torch.float32)

                                if r_eval.model_name == "a2":
                                    scaler_a2 = self.data_pipeline.target_scaler_a2
                                    if scaler_a2 and hasattr(scaler_a2, 'scale_') and scaler_a2.scale_ is not None:
                                        tgt1_t_scaled = torch.tensor(scaler_a2.transform(np.array([[yt_dp]])).flatten(), dtype=torch.float32)
                                        self.online_learner.add_new_sample_for_learning("a2", feats_t, tgt1_t_scaled)
                                    else:
                                        self.logger.warning(f"A2 Scaler 不可用 (ID {r_eval.id})，无法为在线学习缩放目标。")
                                elif r_eval.model_name == "a201":
                                    target1_tensor_logret = torch.tensor([yt_lr if yt_lr is not None else np.nan], dtype=torch.float32)
                                    target2_tensor_raw_dprice = torch.tensor([yt_dp], dtype=torch.float32)
                                    self.online_learner.add_new_sample_for_learning("a201", feats_t, target1_tensor_logret, target2_tensor_raw_dprice)
                            except Exception as e_learn_prep:
                                self.logger.error(f"[错误清扫器] 准备在线学习数据失败 (ID {r_eval.id}): {e_learn_prep}", exc_info=True)

                    if update_params_list_db:
                        with self.db_engine.begin() as conn_update_eval:
                            stmt_upd_eval = text(f"""
                                UPDATE {config.DB_PREDICTION_LOG_TABLE}
                                SET y_true_logret = :ytlr, y_true_dprice = :ytdp, y_true_price  = :ytp,
                                    error_logret  = :elr, error_dprice  = :edp, error_price   = :error_price,
                                    evaluated     = :evaluated_status
                                WHERE id = :id_val
                            """)
                            conn_update_eval.execute(stmt_upd_eval, update_params_list_db)
                        self.logger.info(f"[错误清扫器] 已评估并更新 {len(update_params_list_db)} 条数据库记录。")
            except SQLAlchemyError as e_sql_eval:
                self.logger.error(f"[错误清扫器] 错误评估期间数据库错误：{e_sql_eval}", exc_info=True)
            except Exception as e_eval_generic:
                self.logger.error(f"[错误清扫器] 错误评估期间发生意外错误：{e_eval_generic}", exc_info=True)

            self._stop_event.wait(self.sweep_interval)
        self.logger.info("[错误清扫器] 线程停止。")

    def stop(self):
        self._stop_event.set()
        self.logger.info("[错误清扫器] 已请求停止。")

    def _get_true_price_at_timestamp(self, target_ts: pd.Timestamp, for_model_key: str) -> Optional[float]:
        raw_df = self.data_pipeline.raw_cache.get_market_data_copy()
        price: Optional[float] = None
        if not raw_df.empty and raw_df.index.tz is None:
            raw_df.index = raw_df.index.tz_localize(timezone.utc)
        raw_df = raw_df.sort_index()

        if target_ts.tzinfo is None: target_ts_utc = target_ts.tz_localize(timezone.utc)
        elif target_ts.tzinfo != timezone.utc: target_ts_utc = target_ts.tz_convert(timezone.utc)
        else: target_ts_utc = target_ts

        if not raw_df.empty and 'kline_close' in raw_df.columns:
            try:
                if not raw_df.index.is_monotonic_increasing: raw_df = raw_df.sort_index()
                idx_arr = raw_df.index.get_indexer([target_ts_utc], method='pad')
                if idx_arr.size > 0 and idx_arr[0] != -1:
                    idx_val = idx_arr[0]
                    actual_ts = raw_df.index[idx_val]
                    diff_s = abs((target_ts_utc - actual_ts).total_seconds())
                    k_interval_s = pd.Timedelta(config.A1_FREQ if for_model_key == "a2" else config.A101_AGG_PERIOD).total_seconds()
                    allowed_diff = max(k_interval_s / 2.0, 5.0)
                    if diff_s <= allowed_diff: price = float(raw_df['kline_close'].iloc[idx_val])
            except IndexError: self.logger.warning(f"缓存价格获取发生 IndexError (时间 {target_ts_utc}, 模型 {for_model_key})。")
            except Exception as e: self.logger.warning(f"缓存价格获取错误 (时间 {target_ts_utc}, 模型 {for_model_key}): {e}")

        if price is None:
            k_interval_s = pd.Timedelta(config.A1_FREQ if for_model_key == "a2" else config.A101_AGG_PERIOD).total_seconds()
            q_win_s = int(k_interval_s)
            start_q = target_ts_utc - pd.Timedelta(seconds=q_win_s)
            end_q = target_ts_utc + pd.Timedelta(seconds=q_win_s)
            q_sql = text(f""" SELECT kline_close FROM {config.DB_MARKET_DATA_TABLE} WHERE time_window >= :s AND time_window <= :e ORDER BY ABS(EXTRACT(EPOCH FROM (time_window - :tgt))) LIMIT 1 """)
            try:
                with self.db_engine.connect() as conn:
                    result = conn.execute(q_sql, {"s": start_q, "e": end_q, "tgt": target_ts_utc}).fetchone()
                if result and result[0] is not None: price = float(result[0])
            except Exception as e: self.logger.error(f"数据库价格查询失败 (时间 {target_ts_utc}, 模型 {for_model_key}): {e}")
        return price if price is not None and not pd.isna(price) else None


class OnlineLearningManager:
    def __init__(self, model_service_ref):
        self.logger = logging.getLogger(f"{config.APP_NAME}.OnlineLearningManager")
        self.device = config.DEVICE
        self.model_service = model_service_ref
        self.data_pipeline = self.model_service.data_pipeline_manager
        self.db_engine = self.data_pipeline.engine
        self.model_a2 = self.model_service.model_a2
        self.model_a201 = self.model_service.model_a201
        self.optimizer_a2: Optional[optim.Optimizer] = None
        self.ema_a2: Optional[EMA] = None
        self.criterion_a2_online = None
        if self.model_a2 is not None:
            self.optimizer_a2 = optim.AdamW(self.model_a2.parameters(), lr=config.ONLINE_LR_A2, weight_decay=1e-7)
            self.ema_a2 = EMA(self.model_a2, decay=config.EMA_DECAY_DEFAULT)
            if weighted_loss_a2 is not None: self.criterion_a2_online = weighted_loss_a2
            else: self.logger.error("weighted_loss_a2 未定义；A2 在线学习将失败。")

        self.optimizer_a201: Optional[optim.Optimizer] = None
        self.ema_a201: Optional[EMA] = None
        if self.model_a201 is not None:
            self.optimizer_a201 = optim.AdamW(self.model_a201.parameters(), lr=config.ONLINE_LR_A201, weight_decay=1e-7)
            self.ema_a201 = EMA(self.model_a201, decay=config.EMA_DECAY_DEFAULT)

        self.criterion_logret_a201_online = nn.MSELoss()
        self.criterion_price_delta_a201_online = smooth_l1_a201
        self.replay_buffer_a2: deque = deque(maxlen=config.REPLAY_BUFFER_SIZE)
        self.replay_buffer_a201: deque = deque(maxlen=config.REPLAY_BUFFER_SIZE)
        self.new_data_cache_a2: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.new_data_cache_a201: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.batch_trigger = config.ONLINE_BATCH_TRIGGER
        self.train_cut_ts = {"a2": config.TRAIN_END_TS_A2, "a201": config.TRAIN_END_TS_A201}
        self.last_online_ts = self.train_cut_ts.copy()
        self._refresh_train_cut_ts()
        self.delay_td = pd.Timedelta(minutes=config.ONLINE_LEARNING_DELAY_MINUTES)
        self.stable_model_a2 = self._copy_model_instance(self.model_a2, "a2")
        self.stable_model_a201 = self._copy_model_instance(self.model_a201, "a201")
        self.logger.info(f"在线学习管理器 (OnlineLearningManager) 已初始化，批处理触发条数：{config.ONLINE_BATCH_TRIGGER}")
        threading.Thread(target=self.time_scheduler, daemon=True, name="OLM_TimeScheduler").start()
        #threading.Thread(target=self._bootstrap_from_backlog, daemon=True, name="OLM_BacklogBootstrapper").start()
        self.logger.info("历史数据回填功能 (OLM_BacklogBootstrapper) 已暂时禁用。")
    def _refresh_train_cut_ts(self):
        self.train_cut_ts["a2"] = self.train_cut_ts["a2"] or config.TRAIN_END_TS_A2
        self.train_cut_ts["a201"] = self.train_cut_ts["a201"] or config.TRAIN_END_TS_A201
        self.last_online_ts = self.train_cut_ts.copy()

    def _time_to_update(self, model: str) -> bool:
        last = self.last_online_ts.get(model)
        if last is None:
            self.logger.info(f"模型 {model} 的 last_online_ts 为空，计划更新。")
            return True
        now = pd.Timestamp.utcnow()
        interval = pd.Timedelta(minutes=config.ONLINE_LEARNING_INTERVAL_MINUTES)
        return now >= last + interval

    def _run_time_block_update(self, model: str, start_ts_param: Optional[pd.Timestamp], end_ts_param: pd.Timestamp):
        start_ts = start_ts_param
        end_ts = end_ts_param

        if start_ts is None:
            start_ts = config.TRAIN_END_TS_A2 if model == "a2" else config.TRAIN_END_TS_A201
            if start_ts is None:
                self.logger.warning(f"{model.upper()} 在线学习跳过：train_end_ts 未初始化。")
                return
            self.last_online_ts[model] = start_ts

        if start_ts.tzinfo is None: start_ts = start_ts.tz_localize('UTC')
        if end_ts.tzinfo is None: end_ts = end_ts.tz_localize('UTC')

        lookback_start = start_ts - pd.Timedelta(days=config.ONLINE_LEARNING_LOOKBACK_DAYS)
        self.logger.info(f"执行模型 {model.upper()} 的时间块更新：周期 {start_ts} 至 {end_ts} (回溯数据始于 {lookback_start})")

        df_block = None
        if model == "a2":
            df_block, _ = self.data_pipeline.gen_a2_training_block(start_ts=lookback_start, end_ts=end_ts)
            if df_block is None or len(df_block) < config.ONLINE_MIN_BLOCK_ROWS:
                self.logger.info(f"A2 在线学习跳过，获取数据块失败或行数 {len(df_block) if df_block is not None else 0} < {config.ONLINE_MIN_BLOCK_ROWS}。")
                self.last_online_ts[model] = end_ts
                return

            # 确保 features_a2 和 a2_target_col_scaled 在 data_pipeline 中可用
            # 这部分代码原有逻辑假设了这些属性的存在，但为了健壮性可以加检查
            if not hasattr(self.data_pipeline, 'features_a2') or not self.data_pipeline.features_a2 or \
               not hasattr(self.data_pipeline, 'a2_target_col_scaled') or not self.data_pipeline.a2_target_col_scaled:
                 self.logger.error("A2 在线学习：data_pipeline 中的 features_a2 或 a2_target_col_scaled 未设置。")
                 self.last_online_ts[model] = end_ts
                 return

            required_cols_a2 = self.data_pipeline.features_a2 + [self.data_pipeline.a2_target_col_scaled]
            missing_cols_a2 = [col for col in required_cols_a2 if col not in df_block.columns]
            if missing_cols_a2:
                self.logger.error(f"A2 在线学习：df_block 缺少列：{missing_cols_a2}。")
                self.last_online_ts[model] = end_ts
                return

            x_np = df_block[self.data_pipeline.features_a2].values.astype('float32')
            y_np = df_block[self.data_pipeline.a2_target_col_scaled].values.astype('float32') # 假设这是scaled dprice

            # 将单个样本特征从 (FeatDim,) 扩展为 (SeqLen=1, FeatDim) 以匹配模型期望（如果模型期望序列输入）
            # 或调整为模型期望的输入形状。当前代码假定模型能处理 (SeqLen, FeatDim)
            # 如果 gen_a2_training_block 返回的是滑动窗口数据，那么 x_np 已经是 (NumSamples, SeqLen, FeatDim)
            # 这里假设 gen_a2_training_block 返回的是扁平化的特征，需要按样本处理
            # 但根据 _bootstrap_from_backlog，样本特征是 (SeqLen, FeatDim)
            # 为了统一，这里也应该构造成 (SeqLen, FeatDim) 的样本
            # gen_a2_training_block 应该返回滑动窗口处理后的数据
            # 如果它返回的是 DataFrame，每行是一个时间点，我们需要自己做滑窗
            # 鉴于 _bootstrap_from_backlog 的逻辑，我们这里简化，假设df_block每行是一个样本的最后一行特征
            # 这与_bootstrap_from_backlog中的 (SeqLen, FeatDim) 不一致，需要调整gen_a2_training_block的输出
            # **重要假设**：gen_a2_training_block 返回的 df_block 已经是适合制作 (SeqLen, FeatDim) 输入的了
            # 例如，通过滑动窗口切分。以下代码将df_block的每一行视为一个 (SeqLen, FeatDim) 的样本。
            # 这通常不正确，除非 gen_a2_training_block 做了特殊处理。
            # **修正思路**：gen_a2_training_block应返回可直接用于制作 (N, SeqLen, FeatDim) 的数据。
            # 此处临时处理：将整个 df_block 视为一个序列 (如果长度足够)
            if len(df_block) >= config.A2_SEQ_LEN:
                 x_seq = df_block[self.data_pipeline.features_a2].tail(config.A2_SEQ_LEN).values.astype('float32')
                 # 目标是对应序列的最后一个时间点
                 y_target = df_block[self.data_pipeline.a2_target_col_scaled].iloc[-1] # 对应序列的最后一个目标
                 self.new_data_cache_a2.append(
                     (torch.from_numpy(x_seq), torch.tensor([y_target], dtype=torch.float32))
                 )
                 self.logger.info(f"A2: 从时间块添加 1 个序列样本到 new_data_cache。缓存大小: {len(self.new_data_cache_a2)}")
                 if len(self.new_data_cache_a2) >= self.batch_trigger: self._trigger_update_a2()
            else:
                self.logger.info(f"A2: 时间块数据行数 {len(df_block)} 不足序列长度 {config.A2_SEQ_LEN}。")


        elif model == "a201":
            df_block, _ = self.data_pipeline.gen_a201_training_block(start_ts=lookback_start, end_ts=end_ts)
            if df_block is None or len(df_block) < config.ONLINE_MIN_BLOCK_ROWS:
                self.logger.info(f"A201 在线学习跳过，获取数据块失败或行数 {len(df_block) if df_block is not None else 0} < {config.ONLINE_MIN_BLOCK_ROWS}。")
                self.last_online_ts[model] = end_ts
                return

            if not self.data_pipeline.features_a201:
                self.logger.error("A201 在线学习：data_pipeline 中的 features_a201 未设置。")
                self.last_online_ts[model] = end_ts
                return

            required_cols_a201 = self.data_pipeline.features_a201 + ["target_logret", "target_dprice"] # 假设gen_a201_training_block提供这些目标列
            missing_cols_a201 = [col for col in required_cols_a201 if col not in df_block.columns]
            if missing_cols_a201:
                self.logger.error(f"A201 在线学习：df_block 缺少列：{missing_cols_a201}。")
                self.last_online_ts[model] = end_ts
                return

            # 与A2类似，假设gen_a201_training_block返回适合制作 (N, SeqLen, FeatDim) 的数据
            # 临时处理：
            if len(df_block) >= config.A201_SEQ_LEN:
                x_seq = df_block[self.data_pipeline.features_a201].tail(config.A201_SEQ_LEN).values.astype('float32')
                y1_target = df_block["target_logret"].iloc[-1]
                y2_target = df_block["target_dprice"].iloc[-1] # 这是原始dprice
                self.new_data_cache_a201.append(
                    (torch.from_numpy(x_seq),
                     torch.tensor([y1_target], dtype=torch.float32),
                     torch.tensor([y2_target], dtype=torch.float32))
                )
                self.logger.info(f"A201: 从时间块添加 1 个序列样本到 new_data_cache。缓存大小: {len(self.new_data_cache_a201)}")
                if len(self.new_data_cache_a201) >= self.batch_trigger: self._trigger_update_a201()
            else:
                 self.logger.info(f"A201: 时间块数据行数 {len(df_block)} 不足序列长度 {config.A201_SEQ_LEN}。")


        self.last_online_ts[model] = end_ts
        self.logger.info(f"{model.upper()} 在线学习时间块处理完成，新的 last_online_ts = {end_ts}")


    def time_scheduler(self):
        self.logger.info("在线学习时间调度器已启动。")
        while True:
            now_utc = pd.Timestamp.utcnow()
            for m in ("a2", "a201"):
                if self._time_to_update(m):
                    self.logger.info(f"模型 {m} 到达更新时间。上次更新：{self.last_online_ts.get(m)}")
                    start_for_block = self.last_online_ts.get(m)
                    if start_for_block is None:
                        self.logger.warning(f"模型 {m} 的 last_online_ts 在调度器中仍为空。尝试使用 TRAIN_END_TS。")
                        start_for_block = config.TRAIN_END_TS_A2 if m == "a2" else config.TRAIN_END_TS_A201
                        if start_for_block is None:
                            self.logger.error(f"无法确定模型 {m} 在线学习的开始时间。跳过本次更新。")
                            continue
                    self._run_time_block_update(m, start_for_block, now_utc)
            time.sleep(60)


    def _bootstrap_from_backlog(self):
        # 等待其他组件（尤其是DataPipelineManager和ModelPredictionService）初始化完毕
        time.sleep(15) # 简单延迟，实际应用中可能需要更复杂的依赖管理
        self.logger.info("启动历史积压数据回填...")

        now = pd.Timestamp.utcnow()

        def _fetch_rows(model_name: str, horizon_sec: int):
            start_ts = self.train_cut_ts.get(model_name)
            if start_ts is None:
                self.logger.warning(f"{model_name} 回填跳过：train_cut_ts 未设置。")
                return []
            if start_ts.tzinfo is None: start_ts = start_ts.tz_localize('UTC')

            end_ts_backlog = now - pd.Timedelta(seconds=horizon_sec) - pd.Timedelta(minutes=5)
            if start_ts >= end_ts_backlog:
                 self.logger.info(f"{model_name} 回填：无积压数据时间段 (开始 {start_ts} >= 结束 {end_ts_backlog})。")
                 return []
            self.logger.info(f"{model_name} 回填：获取积压数据自 {start_ts} 至 {end_ts_backlog}")

            sql_adj = f"""
                SELECT raw_features_pickle AS features, y_true_logret, y_true_dprice, ts_pred
                FROM {config.DB_PREDICTION_LOG_TABLE}
                WHERE model_name = :m AND evaluated = TRUE AND ts_pred > :start_pred_ts AND ts_target <= :end_target_ts
                ORDER BY ts_target ASC
            """
            params = {"m": model_name, "start_pred_ts": start_ts, "end_target_ts": end_ts_backlog}
            try:
                with self.db_engine.connect() as conn: return conn.execute(text(sql_adj), params).fetchall()
            except Exception as e:
                self.logger.error(f"{model_name} 获取回填数据失败：{e}", exc_info=True)
                return []

        if self.data_pipeline.features_a2 and self.model_a2:
            expected_dim_a2 = len(self.data_pipeline.features_a2)
            rows_a2 = _fetch_rows("a2", self.data_pipeline.a1_horizon_seconds_for_a2_target)
            a2_samples_added = 0
            for r_idx, (feats_bin, _, y_true_dp_raw) in enumerate(rows_a2):
                if feats_bin is None or y_true_dp_raw is None:
                    self.logger.warning(f"A2 回填：跳过行 {r_idx}，特征或原始目标为空。")
                    continue
                try:
                    with io.BytesIO(feats_bin) as bio:
                        data_loaded = np.load(bio, allow_pickle=True)
                        feats_arr = data_loaded["features_array"]
                    if feats_arr.ndim != 2 or feats_arr.shape[0] != config.A2_SEQ_LEN or feats_arr.shape[1] != expected_dim_a2:
                        self.logger.warning(f"A2 回填跳过 (行 {r_idx})：特征维度 {feats_arr.shape}，期望 ({config.A2_SEQ_LEN}, {expected_dim_a2})")
                        continue
                    x = torch.tensor(feats_arr, dtype=torch.float32)
                    scaler_a2 = self.data_pipeline.target_scaler_a2
                    if scaler_a2 and hasattr(scaler_a2, 'scale_') and scaler_a2.scale_ is not None:
                        y_scaled_dp = scaler_a2.transform(np.array([[y_true_dp_raw]]))[0,0]
                        y = torch.tensor([y_scaled_dp], dtype=torch.float32)
                        self.new_data_cache_a2.append((x, y))
                        a2_samples_added +=1
                    else: self.logger.warning(f"A2 回填：A2 目标缩放器不可用。跳过样本 (行 {r_idx})。")
                except Exception as e_proc_a2: self.logger.error(f"A2 回填：处理行 {r_idx} 错误：{e_proc_a2}", exc_info=True)
            self.logger.info(f"A2 回填：已添加 {a2_samples_added} 个样本到 new_data_cache_a2。")

        if self.data_pipeline.features_a201 and self.model_a201:
            expected_dim_a201 = len(self.data_pipeline.features_a201)
            rows_a201 = _fetch_rows("a201", self.data_pipeline.a201_horizon_seconds)
            a201_samples_added = 0
            for r_idx, (feats_bin, y_true_lr, y_true_dp_raw) in enumerate(rows_a201):
                if feats_bin is None or y_true_lr is None or y_true_dp_raw is None:
                    self.logger.warning(f"A201 回填：跳过行 {r_idx}，特征或目标为空。")
                    continue
                try:
                    with io.BytesIO(feats_bin) as bio:
                        data_loaded = np.load(bio, allow_pickle=True)
                        feats_arr = data_loaded["features_array"]
                    if feats_arr.ndim != 2 or feats_arr.shape[0] != config.A201_SEQ_LEN or feats_arr.shape[1] != expected_dim_a201:
                        self.logger.warning(f"A201 回填跳过 (行 {r_idx})：特征维度 {feats_arr.shape}，期望 ({config.A201_SEQ_LEN}, {expected_dim_a201})")
                        continue
                    x = torch.tensor(feats_arr, dtype=torch.float32)
                    y1 = torch.tensor([y_true_lr], dtype=torch.float32)
                    y2 = torch.tensor([y_true_dp_raw], dtype=torch.float32)
                    self.new_data_cache_a201.append((x, y1, y2))
                    a201_samples_added +=1
                except Exception as e_proc_a201: self.logger.error(f"A201 回填：处理行 {r_idx} 错误：{e_proc_a201}", exc_info=True)
            self.logger.info(f"A201 回填：已添加 {a201_samples_added} 个样本到 new_data_cache_a201。")

        self.logger.info(f"回填完成：a2={len(self.new_data_cache_a2)}, a201={len(self.new_data_cache_a201)} 条样本已添加到新数据缓存。")
        if len(self.new_data_cache_a2) >= self.batch_trigger:
            self.logger.info("回填后触发 A2 更新。")
            self._trigger_update_a2()
        if len(self.new_data_cache_a201) >= self.batch_trigger:
            self.logger.info("回填后触发 A201 更新。")
            self._trigger_update_a201()
        self.logger.info("历史积压数据回填处理完毕。")


    def _copy_model_instance(self, model: Optional[nn.Module], model_key: str) -> Optional[nn.Module]:
        if model is None: return None
        try:
            copied = copy.deepcopy(model).to(self.device)
            copied.eval(); [p.requires_grad_(False) for p in copied.parameters()]
            self.logger.info(f"已通过深拷贝复制 {model_key} 稳定模型。")
            return copied
        except Exception as e_deep: self.logger.warning(f"深拷贝 {model_key} 失败 ({e_deep})，尝试重新实例化…")

        params: Dict[str, Any]
        ModelClsToUse: Optional[Type[nn.Module]] = None
        if model_key == "a2":
            if ModelA2 is None: self.logger.error("ModelA2 类不可用，无法回退复制。"); return None
            if self.model_service.input_dim_a2 is None: self.logger.error("input_dim_a2 为空，无法回退复制。"); return None
            base = self.model_service.a2_loaded_model_config or config.A2_MODEL_CONSTRUCTOR_PARAMS_DEFAULT
            params = base.copy()
            params["input_dim"] = self.model_service.input_dim_a2
            params["use_gpu_wavelet"] = config.HAS_PYTORCH_WAVELETS and config.DEVICE.type == "cuda"
            ModelClsToUse = ModelA2
        elif model_key == "a201":
            if ModelA201 is None: self.logger.error("ModelA201 类不可用，无法回退复制。"); return None
            if self.model_service.input_dim_a201 is None: self.logger.error("input_dim_a201 为空，无法回退复制。"); return None
            base = self.model_service.a201_loaded_model_config or config.A201_MODEL_CONSTRUCTOR_PARAMS_DEFAULT
            params = base.copy()
            params["in_dim"] = self.model_service.input_dim_a201
            ModelClsToUse = ModelA201
        else: self.logger.error(f"未知的 model_key {model_key}，无法回退复制。"); return None

        if ModelClsToUse is None: return None
        try:
            fallback = ModelClsToUse(**params).to(self.device)
            fallback.load_state_dict(model.state_dict())
            fallback.eval(); [p.requires_grad_(False) for p in fallback.parameters()]
            self.logger.info(f"已通过回退方式复制稳定模型 {model_key}。")
            return fallback
        except Exception as e_f: self.logger.error(f"{model_key} 回退复制仍失败: {e_f}", exc_info=True); return None

    def _ready(self, model_key: str, now_ts: pd.Timestamp) -> bool:
        cut = self.train_cut_ts.get(model_key)
        if cut is None:
            self.logger.warning(f"模型 {model_key} 的 train_cut_ts 为空。假定已准备好进行在线学习。")
            return True
        if cut.tzinfo is None: cut = cut.tz_localize(timezone.utc)
        return now_ts >= cut + self.delay_td

    def add_new_sample_for_learning(
            self, model_name: str, features_tensor: torch.Tensor,
            target1_tensor: torch.Tensor, target2_tensor: Optional[torch.Tensor] = None,
    ) -> None:
        now_ts = pd.Timestamp.utcnow()
        if not self._ready(model_name, now_ts):
            self.logger.debug(f"模型 {model_name} 在线学习尚未就绪 (延迟期)。时间 {now_ts} 的样本被忽略。")
            return

        if features_tensor.ndim == 1: pass # (1, FeatureDim) or (FeatureDim) - handle in training loop if needed
        elif features_tensor.ndim == 2: pass # (SeqLen, FeatureDim)
        else: self.logger.error(f"模型 {model_name} 的特征张量维度无效：{features_tensor.ndim}"); return

        if model_name == "a2":
            if self.model_a2 is None or self.criterion_a2_online is None: return
            self.new_data_cache_a2.append((features_tensor.cpu(), target1_tensor.cpu().squeeze()))
            if len(self.new_data_cache_a2) >= self.batch_trigger: self._trigger_update_a2()
        elif model_name == "a201":
            if self.model_a201 is None or target2_tensor is None: return
            self.new_data_cache_a201.append((features_tensor.cpu(), target1_tensor.cpu().squeeze(), target2_tensor.cpu().squeeze()))
            if len(self.new_data_cache_a201) >= self.batch_trigger: self._trigger_update_a201()

    def _train_loop_online(
            self, model: nn.Module, optimizer: optim.Optimizer, ema: Optional[EMA],
            loader: DataLoader, criterion_fn: Optional[Any],
            stable_model: Optional[nn.Module], name: str,
    ) -> None:
        ## 在线学习禁用点 ##
        if not config.ENABLE_ONLINE_LEARNING:
            self.logger.info(f"{name.upper()} 在线学习训练循环被禁用 (通过配置)。")
            return

        if model is None or optimizer is None:
            self.logger.error(f"模型 {name} 在线训练循环无法运行：模型或优化器为空。")
            return
        if name == "a2" and criterion_fn is None:
            self.logger.error(f"A2 训练循环被调用，但模型 {name} 的 criterion_fn 为空。")
            return

        model.train()
        for epoch_idx in range(config.ONLINE_LEARNING_EPOCHS):
            total_loss, sample_cnt = 0.0, 0
            for batch_data in loader:
                x_b, y1_b = batch_data[0].to(self.device), batch_data[1].to(self.device)
                optimizer.zero_grad()
                loss = torch.tensor(0.0, device=self.device)

                if name == "a2":
                    pred_b = model(x_b, None)
                    loss = criterion_fn(pred_b, y1_b.view_as(pred_b)).mean()
                elif name == "a201":
                    y2_b = batch_data[2].to(self.device)
                    pred_lr_b, pred_dp_raw_b = model(x_b)
                    pred_dp_enc_b = dprice_encode_a201(pred_dp_raw_b)
                    true_dp_enc_b = dprice_encode_a201(y2_b.view_as(pred_dp_raw_b))
                    loss_dp_b = self.criterion_price_delta_a201_online(pred_dp_enc_b, true_dp_enc_b)
                    loss_lr_b = self.criterion_logret_a201_online(pred_lr_b, y1_b.view_as(pred_lr_b))
                    loss = config.LOSS_WEIGHT_DELTA_PRICE * loss_dp_b + config.LOSS_WEIGHT_LOGRET * loss_lr_b
                else: self.logger.error(f"在线训练循环中未知的模型名称 '{name}'。"); continue

                if stable_model and config.EWC_LAMBDA > 0:
                    ewc = config.EWC_LAMBDA * sum(torch.sum((p - ps.detach()) ** 2) for p, ps in zip(model.parameters(), stable_model.parameters()) if p.requires_grad)
                    loss += ewc
                if stable_model and config.ANCHOR_LAMBDA > 0:
                    anchor = config.ANCHOR_LAMBDA * sum(torch.sum((p - ps.detach()) ** 2) for p, ps in zip(model.parameters(), stable_model.parameters()) if p.requires_grad)
                    loss += anchor

                loss.backward(); optimizer.step()
                if ema: ema.update(model)
                total_loss += loss.item() * x_b.size(0); sample_cnt += x_b.size(0)
            avg_loss = total_loss / sample_cnt if sample_cnt else float("nan")
            self.logger.info(f"{name.upper()} 在线学习轮次 {epoch_idx + 1}/{config.ONLINE_LEARNING_EPOCHS}，平均损失：{avg_loss:.6f}")
        model.eval()

    def _save_model_weights(self, model: nn.Module, base_path: Path, tag: str) -> None:
        if model is None: self.logger.warning(f"模型 {tag} 为空，跳过保存。"); return

        ts = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
        tmp_path = config.ARTIFACTS_DIR / f"tmp_{tag}_{ts}.pth"
        backup_path = config.ARTIFACTS_DIR / f"online_backup_{tag}_{ts}.pth"
        saved_base = False

        try: torch.save(model.state_dict(), tmp_path)
        except Exception as e: self.logger.error(f"保存临时文件 {tmp_path} (模型 {tag}) 失败：{e}"); return

        try:
            base_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(tmp_path, base_path); saved_base = True
            self.logger.info(f"{tag} 权重已原子性保存至 {base_path}")
        except OSError as e_replace:
            self.logger.warning(f"原子替换至 {base_path} 失败 ({e_replace})，尝试直接保存。")
            try:
                torch.save(model.state_dict(), base_path); saved_base = True
                self.logger.info(f"直接保存至 {base_path} 成功。")
                if tmp_path.exists(): tmp_path.unlink(missing_ok=True)
            except Exception as e_direct: self.logger.error(f"直接保存至 {base_path} (模型 {tag}) 仍失败：{e_direct}")
        finally:
            if tmp_path.exists(): tmp_path.unlink(missing_ok=True)

        if saved_base:
            try:
                config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), backup_path)
                self.logger.info(f"{tag} 权重也已备份至 {backup_path}")
                backups = sorted(config.ARTIFACTS_DIR.glob(f"online_backup_{tag}_*.pth"))
                for p_backup in backups[:-6]: # 保留最近6个备份
                    try: p_backup.unlink(missing_ok=True); self.logger.debug(f"已删除旧备份：{p_backup}")
                    except Exception as e_del_backup: self.logger.warning(f"删除旧备份 {p_backup} 失败：{e_del_backup}")
            except Exception as e_backup_save: self.logger.error(f"保存备份文件 {backup_path} (模型 {tag}) 失败：{e_backup_save}")

    def _trigger_update_a2(self):
        if not self.new_data_cache_a2: self.logger.debug("A2：新数据缓存为空，不触发更新。"); return
        if self.model_a2 is None or self.optimizer_a2 is None or self.criterion_a2_online is None:
            self.logger.warning("A2：模型、优化器或损失函数未就绪。跳过在线更新触发。")
            self.replay_buffer_a2.extend(self.new_data_cache_a2); self.new_data_cache_a2.clear()
            return

        self.logger.info(f"A2 在线学习触发：新数据缓存={len(self.new_data_cache_a2)}")
        train_data = list(self.new_data_cache_a2)
        if self.replay_buffer_a2 and config.REPLAY_RATIO > 0 and train_data:
            k = int(min(len(self.replay_buffer_a2), len(train_data) * config.REPLAY_RATIO))
            if k > 0: train_data.extend(random.sample(list(self.replay_buffer_a2), k))
        if not train_data: self.logger.info("A2：与回放缓冲区合并后无数据可训练。"); self.new_data_cache_a2.clear(); return
        random.shuffle(train_data)
        xs, ys = zip(*train_data)
        try:
            xs_stacked = torch.stack(xs); ys_stacked = torch.stack(ys).squeeze()
            if ys_stacked.ndim == 0: ys_stacked = ys_stacked.unsqueeze(0)
        except Exception as e_stack:
            self.logger.error(f"A2：为 DataLoader 堆叠张量失败：{e_stack}。数据形状：{[x.shape for x in xs[:2]]}, {[y.shape for y in ys[:2]]}")
            self.new_data_cache_a2.clear(); return

        loader = DataLoader(
            TensorDataset(xs_stacked, ys_stacked), batch_size=min(config.ONLINE_BATCH_SIZE, len(train_data)),
            shuffle=True, pin_memory=(self.device.type == 'cuda')
        )

        if config.ENABLE_ONLINE_LEARNING:
            self._train_loop_online(self.model_a2, self.optimizer_a2, self.ema_a2, loader, self.criterion_a2_online, self.stable_model_a2, "a2")
            if self.ema_a2: self.ema_a2.apply_shadow(self.model_a2)
            self._save_model_weights(self.model_a2, config.A2_BEST_MODEL_PATH, "a2_online_update")
            if hasattr(self.model_service, "reload_weights"):
                try:
                    self.model_service.reload_weights("a2", config.A2_BEST_MODEL_PATH)
                    meta_path_a2 = config.A2_BEST_MODEL_PATH.with_suffix(config.A2_BEST_MODEL_PATH.suffix + '.meta_online')
                    json.dump({"online_update_timestamp": pd.Timestamp.utcnow().isoformat()}, open(meta_path_a2, "w"))
                    self.logger.info("A2 热更新完成 ✅")
                except Exception as e: self.logger.warning(f"A2 权重热更新失败：{e}")
            new_stable_a2 = self._copy_model_instance(self.model_a2, "a2")
            if new_stable_a2: self.stable_model_a2 = new_stable_a2
        else: self.logger.info("A2 在线学习训练本身被禁用 (通过配置)。")

        self.replay_buffer_a2.extend(self.new_data_cache_a2); self.new_data_cache_a2.clear()

    def _trigger_update_a201(self):
        if not self.new_data_cache_a201: self.logger.debug("A201：新数据缓存为空，不触发更新。"); return
        if self.model_a201 is None or self.optimizer_a201 is None:
            self.logger.warning("A201：模型或优化器未就绪。跳过在线更新触发。")
            self.replay_buffer_a201.extend(self.new_data_cache_a201); self.new_data_cache_a201.clear()
            return

        self.logger.info(f"A201 在线学习触发：新数据缓存={len(self.new_data_cache_a201)}")
        train_data = list(self.new_data_cache_a201)
        if self.replay_buffer_a201 and config.REPLAY_RATIO > 0 and train_data:
            k = int(min(len(self.replay_buffer_a201), len(train_data) * config.REPLAY_RATIO))
            if k > 0: train_data.extend(random.sample(list(self.replay_buffer_a201), k))
        if not train_data: self.logger.info("A201：与回放缓冲区合并后无数据可训练。"); self.new_data_cache_a201.clear(); return
        random.shuffle(train_data)
        xs, ys_lr, ys_dp_raw = zip(*train_data)
        try:
            xs_stacked = torch.stack(xs); ys_lr_stacked = torch.stack(ys_lr).squeeze(); ys_dp_raw_stacked = torch.stack(ys_dp_raw).squeeze()
            if ys_lr_stacked.ndim == 0: ys_lr_stacked = ys_lr_stacked.unsqueeze(0)
            if ys_dp_raw_stacked.ndim == 0: ys_dp_raw_stacked = ys_dp_raw_stacked.unsqueeze(0)
        except Exception as e_stack:
            self.logger.error(f"A201：为 DataLoader 堆叠张量失败：{e_stack}。数据形状：{[x.shape for x in xs[:2]]}, {[y.shape for y in ys_lr[:2]]}")
            self.new_data_cache_a201.clear(); return

        loader = DataLoader(
            TensorDataset(xs_stacked, ys_lr_stacked, ys_dp_raw_stacked), batch_size=min(config.ONLINE_BATCH_SIZE, len(train_data)),
            shuffle=True, pin_memory=(self.device.type == 'cuda')
        )

        if config.ENABLE_ONLINE_LEARNING:
            self._train_loop_online(self.model_a201, self.optimizer_a201, self.ema_a201, loader, None, self.stable_model_a201, "a201")
            if self.ema_a201: self.ema_a201.apply_shadow(self.model_a201)
            self._save_model_weights(self.model_a201, config.A201_BEST_MODEL_PATH, "a201_online_update")
            if hasattr(self.model_service, "reload_weights"):
                try:
                    self.model_service.reload_weights("a201", config.A201_BEST_MODEL_PATH)
                    meta_path_a201 = config.A201_BEST_MODEL_PATH.with_suffix(config.A201_BEST_MODEL_PATH.suffix + '.meta_online')
                    json.dump({"online_update_timestamp": pd.Timestamp.utcnow().isoformat()}, open(meta_path_a201, "w"))
                    self.logger.info("A201 热更新完成 ✅")
                except Exception as e: self.logger.warning(f"A201 权重热更新失败：{e}")
            new_stable_a201 = self._copy_model_instance(self.model_a201, "a201")
            if new_stable_a201: self.stable_model_a201 = new_stable_a201
        else: self.logger.info("A201 在线学习训练本身被禁用 (通过配置)。")

        self.replay_buffer_a201.extend(self.new_data_cache_a201); self.new_data_cache_a201.clear()

    def training_daemon(self):
        self.logger.info("在线学习守护进程 (training_daemon) 已启动。注意：主要调度由 time_scheduler 执行。")
        while True:
            if len(self.new_data_cache_a2) >= self.batch_trigger:
                self.logger.info(f"守护进程：A2 缓存达到触发大小 ({len(self.new_data_cache_a2)} >= {self.batch_trigger})。触发更新。")
                self._trigger_update_a2()
            if len(self.new_data_cache_a201) >= self.batch_trigger:
                self.logger.info(f"守护进程：A201 缓存达到触发大小 ({len(self.new_data_cache_a201)} >= {self.batch_trigger})。触发更新。")
                self._trigger_update_a201()
            time.sleep(5)