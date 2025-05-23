#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版 v3.10 a101.py — BTC/USDT 3 min 数据预处理 (对数收益率目标)
--------------------------------------------------
(改动 v3.12.1 - 修正 normalize_features 遗漏，并整合所有讨论的修改)
- 补全 normalize_features 函数的完整定义。
- get_db_latest_timestamp 和 load_data_from_db 修改为接收 Engine 对象。
- preprocess_data 添加 db_engine_override 参数，优先使用传入引擎。
- 使用模块级命名的 logger (logger_a101)。
- 移除了文件级的 basicConfig，应由主程序统一配置。
- 修正了 _calculate_refined_roc 中的 Inf 处理。
- 优化了 resample_and_merge 中列名处理。
- 在 add_features 中集成 Inf/NaN 检查和初步处理。
- 在 normalize_features 中集成健壮的 Inf/NaN 清理和 scaler 兼容性检查。
"""
#from config import SCALER_PATH_DEFAULT
# 然后在代码里直接用 SCALER_PATH_DEFAULT
import config

import logging
from pathlib import Path
from typing import Tuple, List, Optional
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text, Engine
from config import MIN_CHAIN_DATA_HISTORY_FOR_A101_STR, A101_AGG_PERIOD

try:
    from ta.volatility import AverageTrueRange, BollingerBands
    from ta.momentum import RSIIndicator
    from ta.trend import MACD
    from ta.volume import OnBalanceVolumeIndicator

    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

try:
    from zuixinmoxing.jinping.config import ARTIFACTS_DIR, LAG_STEPS as CONFIG_LAG_STEPS
except ImportError:
    ARTIFACTS_DIR = Path("./a101_artifacts_fallback")  # Fallback if config not found
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_LAG_STEPS = [1, 2, 3, 5, 10, 20]
    # Conditional logging if logger is already configured by main app
    # if logging.getLogger("TradingSystem.A101Preprocessor").hasHandlers():
    #     logging.getLogger("TradingSystem.A101Preprocessor").warning(
    #         "a101.py: Cannot import ARTIFACTS_DIR, LAG_STEPS from main config. Using fallback values."
    #     )

AGGREGATION_PERIOD = "3min"
ROLL_WINDOW: int = 15
LAG_STEPS: List[int] = CONFIG_LAG_STEPS

EXCLUDE_COLS = {
    "id", "inserted_at", "height", "mempool_size",
    "total_mempool_value", "avg_mempool_fee_rate",
}
DB_HOST, DB_PORT, DB_NAME = "localhost", 5432, "crypto_data"
DB_USER, DB_PASS = "postgres", "456258"
MARKET_TABLE = "crypto_data"
CHAIN_TABLE = "blockchain_features_utc"
DATABASE_URL_FALLBACK = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

logger_a101 = logging.getLogger("TradingSystem.A101Preprocessor")

ROC_NAN_FILL_VALUE = 0.0  # Changed from -999.0 to 0.0 as NaN/Inf are better handled now
PRED_HORIZON_K_CANDLES_DEFAULT = 1
TARGET_COLUMN_NAME = "future_log_return"


def _safe_division(numerator: pd.Series, denominator: pd.Series, default_value=np.nan) -> pd.Series:
    result = pd.Series(default_value, index=numerator.index, dtype=float)
    valid_mask = denominator.notna() & (denominator.abs() > 1e-9)
    if valid_mask.any():
        result.loc[valid_mask] = numerator.loc[valid_mask] / denominator.loc[valid_mask]
    return result


def _check_and_log_feature_issues(df: pd.DataFrame, feature_names: List[str], step_name: str, logger_ref):
    if not isinstance(feature_names, list): feature_names = [feature_names]
    for feat_name in feature_names:
        if feat_name not in df.columns: continue
        col_data = df[feat_name]
        if col_data.empty: continue
        if pd.api.types.is_numeric_dtype(col_data.dtype):
            inf_mask_series = np.isinf(col_data)
            if inf_mask_series.any():
                inf_count = inf_mask_series.sum()
                logger_ref.error(f"严重：特征 '{feat_name}' (步骤 '{step_name}') 含 {inf_count} 个 Inf！将替换为NaN。")
                df.loc[inf_mask_series, feat_name] = np.nan


def get_db_latest_timestamp(engine: Engine, table_name: str, time_column: str = "time_window") -> Optional[
    pd.Timestamp]:
    try:
        with engine.connect() as connection:
            query = text(f"SELECT MAX({time_column}) AS latest_time FROM {table_name};")
            result = connection.execute(query).fetchone()
        if result and result[0] is not None:
            latest_time_value = result[0]
            try:
                latest_ts = pd.to_datetime(latest_time_value, utc=True)
                logger_a101.debug(f"DB table '{table_name}' latest time: {latest_ts}")
                return latest_ts
            except Exception as e_parse:
                logger_a101.error(f"Error parsing timestamp '{latest_time_value}': {e_parse}")
                return None
        else:
            logger_a101.warning(f"No result or null latest time from DB table '{table_name}'.")
            return None
    except Exception as e:
        logger_a101.error(f"Error querying latest time ('{table_name}'.'{time_column}') using provided engine: {e}")
        return None


def load_data_from_db(engine: Engine, start_time_str: str, end_time_str: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger_a101.info(f"使用共享的 DB 引擎，加载数据区间：{start_time_str} → {end_time_str}")

    market_sql = f""" SELECT time_window AS timestamp, kline_open, kline_high, kline_low, kline_close, kline_volume AS volume, spread_mean, ask_depth_mean, bid_depth_mean, imbalance_mean FROM {MARKET_TABLE} WHERE time_window >= :start_time AND time_window <= :end_time ORDER BY time_window; """
    chain_sql = f"SELECT timestamp, exchange_inflow, exchange_outflow, exchange_netflow, active_addresses, new_addresses, whale_to_exchange, whale_from_exchange, whale_non_exchange, tx_count, difficulty, size_utilization, mean_fee_rate, large_tx_count FROM {CHAIN_TABLE} WHERE timestamp >= :start_time AND timestamp <= :end_time ORDER BY timestamp;"
    market_df, chain_df = pd.DataFrame(), pd.DataFrame()
    try:
        params_sql = {'start_time': start_time_str, 'end_time': end_time_str}
        market_df = pd.read_sql(text(market_sql), engine, params=params_sql)
        logger_a101.info(f"行情数据加载完毕：{len(market_df)} 条")
        chain_df = pd.read_sql(text(chain_sql), engine, params=params_sql)
        logger_a101.info(f"链上特征加载完毕：{len(chain_df)} 条")
    except Exception as e:
        logger_a101.error(f"DB query failed (using provided engine): {e}", exc_info=True)

    for df_obj, df_name_str in [(market_df, "Market"), (chain_df, "Chain")]:
        if not df_obj.empty:
            if "timestamp" not in df_obj.columns: logger_a101.error(
                f"{df_name_str} data missing 'timestamp'."); continue
            df_obj["timestamp"] = pd.to_datetime(df_obj["timestamp"], utc=True, errors='coerce')
            df_obj.dropna(subset=['timestamp'], inplace=True)
            if not df_obj.empty: df_obj.set_index("timestamp", inplace=True); df_obj.sort_index(inplace=True)
    if not chain_df.empty:
        cols_to_drop = [c for c in chain_df.columns if c in EXCLUDE_COLS]
        if cols_to_drop: chain_df = chain_df.drop(columns=cols_to_drop, errors="ignore"); logger_a101.info(
            f"Dropped from chain data: {cols_to_drop}")
    return market_df, chain_df


def resample_and_merge(market_df: pd.DataFrame, chain_df: pd.DataFrame, agg_period: str) -> pd.DataFrame:
    if market_df.empty: logger_a101.warning(
        f"Market data empty, cannot resample to {agg_period}."); return pd.DataFrame()
    if not isinstance(market_df.index, pd.DatetimeIndex): logger_a101.error(
        "Market data index not DatetimeIndex."); return pd.DataFrame()
    logger_a101.info(f"正在将行情数据重采样为{agg_period}K线…")
    market_agg_dict = {}
    if "kline_open" in market_df.columns: market_agg_dict["open"] = market_df["kline_open"].resample(agg_period).first()
    if "kline_high" in market_df.columns: market_agg_dict["high"] = market_df["kline_high"].resample(agg_period).max()
    if "kline_low" in market_df.columns: market_agg_dict["low"] = market_df["kline_low"].resample(agg_period).min()
    if "kline_close" in market_df.columns: market_agg_dict["close"] = market_df["kline_close"].resample(
        agg_period).last()
    if "volume" in market_df.columns: market_agg_dict["volume"] = market_df["volume"].resample(agg_period).sum()
    micro_features_to_agg = ["spread_mean", "ask_depth_mean", "bid_depth_mean", "imbalance_mean"]
    for feat in micro_features_to_agg:
        if feat in market_df.columns: market_agg_dict[feat] = market_df[feat].resample(agg_period).mean()
    market_resampled = pd.DataFrame(market_agg_dict)
    if 'close' in market_resampled.columns:
        market_resampled.dropna(subset=['close'], inplace=True)
    else:
        logger_a101.error(f"Resampled df missing 'close' column."); return pd.DataFrame()
    if market_resampled.empty: logger_a101.warning(
        f"Market data resampled to {agg_period} is empty."); return pd.DataFrame()
    merged_df = market_resampled
    if not chain_df.empty:
        if isinstance(chain_df.index, pd.DatetimeIndex):
            chain_aligned = chain_df.reindex(market_resampled.index, method='ffill', limit=ROLL_WINDOW * 4)
            merged_df = market_resampled.join(chain_aligned, how="left")
        else:
            logger_a101.warning("链上数据索引不是 DatetimeIndex，跳过合并。")
    logger_a101.info(f"重采样（{agg_period}）并合并完成，结果形状：{merged_df.shape}")
    return merged_df


def _calculate_refined_roc(series: pd.Series, prev_series: pd.Series) -> pd.Series:
    roc_series = pd.Series(np.nan, index=series.index, dtype=float)
    both_valid_mask = series.notna() & prev_series.notna()
    prev_zero_mask = both_valid_mask & (np.abs(prev_series) < 1e-9)
    current_also_zero_mask = prev_zero_mask & (np.abs(series) < 1e-9)
    roc_series.loc[current_also_zero_mask] = 0.0
    current_not_zero_from_zero_mask = prev_zero_mask & (np.abs(series) >= 1e-9)
    roc_series.loc[
        current_not_zero_from_zero_mask] = np.nan  # Change from 0 to non-0 is undefined/Inf for ROC, set to NaN
    prev_not_zero_mask = both_valid_mask & (np.abs(prev_series) >= 1e-9)
    roc_series.loc[prev_not_zero_mask] = (series.loc[prev_not_zero_mask] - prev_series.loc[prev_not_zero_mask]) / \
                                         prev_series.loc[prev_not_zero_mask]
    return roc_series


def generate_log_return_target(df_segment: pd.DataFrame, pred_horizon_k: int, price_col_base: str = 'close',
                               price_col_future: str = 'close') -> pd.DataFrame:
    if price_col_base not in df_segment.columns or price_col_future not in df_segment.columns:
        logger_a101.error(f"Price columns for target gen missing.");
        return df_segment.assign(**{TARGET_COLUMN_NAME: np.nan})
    df_labeled = df_segment.copy()
    if len(df_labeled) <= pred_horizon_k:
        df_labeled[TARGET_COLUMN_NAME] = np.nan;
        return df_labeled
    current_price = df_labeled[price_col_base]
    future_price = df_labeled[price_col_future].shift(-pred_horizon_k)
    valid_mask = (current_price.abs() > 1e-9) & future_price.notna() & (future_price.abs() > 1e-9)
    log_returns = pd.Series(np.nan, index=df_labeled.index, name=TARGET_COLUMN_NAME)
    if valid_mask.any(): log_returns.loc[valid_mask] = np.log(
        future_price.loc[valid_mask] / current_price.loc[valid_mask])
    df_labeled[TARGET_COLUMN_NAME] = log_returns
    logger_a101.info(f"已生成 '{TARGET_COLUMN_NAME}' (K={pred_horizon_k})，空值计数：{df_labeled[TARGET_COLUMN_NAME].isnull().sum()}/{len(df_labeled)}")

    return df_labeled


# ============================================================================
# normalize_features 函数定义 - 这是之前遗漏的部分
# ============================================================================
def normalize_features(
        df: pd.DataFrame,
        scaler_save_path: Path
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    仅在推理阶段使用训练期生成的 scaler。
    若列维度或顺序不一致，立即抛错终止，绝不重新拟合。
    """
    current_logger = logger_a101
    if df.empty:
        current_logger.warning("normalize_features: 传入 DataFrame 为空")
        return df, StandardScaler()

    df_s = df.copy()

    # -- 保留原始收盘价与 ATR（若不存在则填 NaN）
    df_s["close_raw"] = df_s.get("close", np.nan)
    df_s["atr_raw"]   = df_s.get("atr",   np.nan)

    cols_not_to_scale = [TARGET_COLUMN_NAME, "close_raw", "atr_raw"]
    numeric_cols_all  = df_s.select_dtypes(include=np.number).columns.tolist()
    numeric_cols_to_scale = [c for c in numeric_cols_all if c not in cols_not_to_scale]

    if not scaler_save_path.exists():
        current_logger.error(f"❌ 未找到训练期 scaler：{scaler_save_path}")
        raise FileNotFoundError(f"Scaler not found: {scaler_save_path}")

    # 加载 scaler
    try:
        scaler: StandardScaler = joblib.load(scaler_save_path)
    except Exception as e:
        current_logger.error(f"加载 scaler 失败: {e}")
        raise RuntimeError("A101_Scaler_Load_Failed") from e

    # 检查列维度与顺序
    # —— 只校验“训练用”那 60 列都在当前 DataFrame 中 ——
    saved_order = list(getattr(scaler, "feature_names_in_", []))
    missing = set(saved_order) - set(numeric_cols_to_scale)
    if missing:
        current_logger.error(f"❌ 预处理缺少训练期特征: {missing}")
        raise RuntimeError("A101_Scaler_Incompatible")  # 要求这些列必须存在

    # —— 仅对这 60 列做标准化 transform ——
    df_s.loc[:, saved_order] = scaler.transform(df_s[saved_order].astype(float))
    setattr(scaler, "_numeric_cols_scaled_by_instance", saved_order)

    # —— 最终只保留这 60 列（以及 close_raw, atr_raw, target 列） ——
    cols_to_keep = saved_order + ["close_raw", "atr_raw", TARGET_COLUMN_NAME]
    df_out = df_s[cols_to_keep].copy()

    return df_out, scaler

def compute_features(df_merge: pd.DataFrame) -> pd.DataFrame:
    """
    将 resample+merge 完成的原始 3min 数据，
    1) 生成 future_log_return
    2) 做好所有衍生特征
    3) 严格调用训练期 scaler 标准化
    返回：只含模型输入特征的 DataFrame，index=name='time_window'。
    抛错：若 scaler 不匹配或缺失列，立即抛异常，保证数据与训练时 1:1 对齐。
    """
    # —— 1. 生成目标列 —— #
    df_labeled = generate_log_return_target(
        df_merge,
        pred_horizon_k=PRED_HORIZON_K_CANDLES_DEFAULT,
        price_col_base='close',
        price_col_future='close'
    )
    if df_labeled.empty or 'future_log_return' not in df_labeled.columns:
        raise RuntimeError("compute_features: 目标列生成失败！")

    # —— 2. 生成衍生特征 —— #
    chain_cols = [c for c in df_merge.columns]  # merge 时带入的所有链上原始列
    df_feat = add_features(
        df_labeled,
        chain_cols_original_names=chain_cols,
        agg_period=A101_AGG_PERIOD,
        current_roll_window=ROLL_WINDOW,
        is_training=False
    )

    if df_feat.empty:
        raise RuntimeError("compute_features: 特征工程失败，add_features 返回空！")

    # —— 3. 标准化 —— #
    # 使用训练时保存的同名 scaler
    scaler_path = Path(ARTIFACTS_DIR) / f"scaler_{A101_AGG_PERIOD}_rw{ROLL_WINDOW}.joblib"
    df_normed, _ = normalize_features(df_feat, scaler_save_path=scaler_path)
    if df_normed.empty:
        raise RuntimeError("compute_features: normalize_features 返回空！")

    # —— 4. 只保留模型输入特征 —— #
    # normalize_features 会留下：
    #   - 所有训练期 scaler 执行 transform 的那几列（特征输入）
    #   - close_raw, atr_raw, future_log_return
    # 我们这里只要输入特征列，剔除 'future_log_return'
    cols = list(df_normed.columns)
    if 'future_log_return' in cols:
        cols.remove('future_log_return')
    # 也可以排除 close_raw, atr_raw，如果模型训练时没用到
    for drop in ('close_raw', 'atr_raw'):
        if drop in cols and drop not in df_normed.columns:
            cols.remove(drop)
    df_final = df_normed[cols].copy()

    # —— 5. 确保 index 名称 —— #
    if isinstance(df_final.index, pd.DatetimeIndex):
        df_final.index.name = 'time_window'

    return df_final
# ============================================================================

def add_features(df: pd.DataFrame, chain_cols_original_names: List[str], agg_period: str,
                 current_roll_window: int, is_training: bool = True) -> pd.DataFrame:
    # This is the version from your v3.10 code, with _check_and_log_feature_issues and _safe_division integrated
    # where appropriate.
    current_logger = logger_a101
    if df.empty: current_logger.warning("Input DataFrame empty for feature engineering."); return df
    current_logger.info(
        f"开始特征工程： {df.shape[0]} 行, {df.shape[1]} 列（聚合： {agg_period}, 滚动窗口： {current_roll_window}).")
    data = df.copy()
    new_features_dict = {}  # Use dict to build features, then concat

    price_col, volume_col, high_col, low_col = "close", "volume", "high", "low"

    # Log Return
    if price_col in data:
        close_for_logret = data[price_col].replace(0, np.nan)
        shifted_close_logret = close_for_logret.shift(1)
        price_ratio_logret = _safe_division(close_for_logret, shifted_close_logret)
        new_features_dict['log_ret'] = np.log(price_ratio_logret.replace(0, np.nan))
        _check_and_log_feature_issues(pd.DataFrame({'log_ret': new_features_dict['log_ret']}), ['log_ret'],
                                      "Log Return Calc", current_logger)

    # Lag Features
    market_cols_for_lag = ['open', 'high', 'low', 'close', 'volume', 'spread_mean', 'ask_depth_mean', 'bid_depth_mean',
                           'imbalance_mean']
    market_cols_present = [col for col in market_cols_for_lag if col in data.columns]
    temp_lag_feats_for_check = []
    for lag in LAG_STEPS:
        for mcol in market_cols_present:
            new_features_dict[f'{mcol}_lag{lag}'] = data[mcol].shift(lag)
            temp_lag_feats_for_check.append(f'{mcol}_lag{lag}')
        if 'log_ret' in new_features_dict:
            new_features_dict[f'log_ret_lag{lag}'] = pd.Series(new_features_dict['log_ret']).shift(lag)
            temp_lag_feats_for_check.append(f'log_ret_lag{lag}')
    if temp_lag_feats_for_check: _check_and_log_feature_issues(
        pd.DataFrame(new_features_dict)[temp_lag_feats_for_check], temp_lag_feats_for_check, "Lag Features",
        current_logger)

    # TA-Lib Features (ATR, RSI, MACD, OBV from your v3.10)
    temp_talib_feats_for_check = []
    atr_window = current_roll_window
    if TALIB_AVAILABLE:
        if all(c in data for c in [high_col, low_col, price_col]):
            try:
                atr_obj = AverageTrueRange(high=data[high_col], low=data[low_col], close=data[price_col],
                                           window=atr_window, fillna=False)
                new_features_dict['atr_raw'] = atr_obj.average_true_range()
                new_features_dict['atr'] = new_features_dict['atr_raw'].copy()
                temp_talib_feats_for_check.extend(['atr_raw', 'atr'])
            except Exception as e:
                current_logger.warning(f"ATR calc error: {e}")
        if price_col in data:
            try:
                new_features_dict['rsi'] = RSIIndicator(close=data[price_col], window=current_roll_window,
                                                        fillna=False).rsi()
                temp_talib_feats_for_check.append('rsi')
                macd_obj = MACD(close=data[price_col], window_slow=26, window_fast=12, window_sign=9, fillna=False)
                new_features_dict['macd'] = macd_obj.macd()
                new_features_dict['macd_signal'] = macd_obj.macd_signal()
                new_features_dict['macd_diff'] = macd_obj.macd_diff()
                temp_talib_feats_for_check.extend(['macd', 'macd_signal', 'macd_diff'])
            except Exception as e:
                current_logger.warning(f"RSI/MACD calc error: {e}")
        if volume_col in data and price_col in data:
            try:
                new_features_dict['obv'] = OnBalanceVolumeIndicator(close=data[price_col], volume=data[volume_col],
                                                                    fillna=False).on_balance_volume()
                temp_talib_feats_for_check.append('obv')
            except Exception as e:
                current_logger.warning(f"OBV calc error: {e}")
    if temp_talib_feats_for_check: _check_and_log_feature_issues(
        pd.DataFrame(new_features_dict)[temp_talib_feats_for_check], temp_talib_feats_for_check, "TA-Lib Features",
        current_logger)

    # Realized Volatility
    if 'log_ret' in new_features_dict:
        series_lr = pd.Series(new_features_dict['log_ret'])
        new_features_dict['realized_vol_rw'] = series_lr.rolling(window=current_roll_window, min_periods=max(1,
                                                                                                             current_roll_window // 2)).std() * np.sqrt(
            current_roll_window)
        _check_and_log_feature_issues(pd.DataFrame({'realized_vol_rw': new_features_dict['realized_vol_rw']}),
                                      ['realized_vol_rw'], "Realized Vol", current_logger)

    # SMA and Price vs SMA
    temp_sma_feats_for_check = []
    if price_col in data:
        for N in [5, 10, current_roll_window, current_roll_window * 2]:
            sma_col = f'sma{N}'
            vs_sma_col = f'price_vs_{sma_col}'
            current_sma_series = data[price_col].rolling(window=N, min_periods=1).mean()
            new_features_dict[sma_col] = current_sma_series
            new_features_dict[vs_sma_col] = _safe_division(data[price_col], current_sma_series,
                                                           default_value=1.0) - 1.0  # if sma is 0, ratio is 1, result 0
            temp_sma_feats_for_check.extend([sma_col, vs_sma_col])
    if temp_sma_feats_for_check: _check_and_log_feature_issues(
        pd.DataFrame(new_features_dict)[temp_sma_feats_for_check], temp_sma_feats_for_check, "SMA Features",
        current_logger)
    MIN_UNIQUE_VALUES_FOR_DERIVATIVES = 2
    MIN_STD_FOR_DERIVATIVES = 1e-9
    # Chain Features Derivatives
    if is_training:
        chain_cols_to_derive = []
        for col_name_iter in chain_cols_original_names:
            if col_name_iter in data.columns:
                series_to_check = data[col_name_iter].dropna()
                if not series_to_check.empty and series_to_check.nunique() >= MIN_UNIQUE_VALUES_FOR_DERIVATIVES \
                        and (series_to_check.std() if len(series_to_check) > 1 else 0.0) >= MIN_STD_FOR_DERIVATIVES:
                    chain_cols_to_derive.append(col_name_iter)
    else:
        chain_cols_to_derive = [c for c in chain_cols_original_names if c in data.columns]

    temp_chain_deriv_feats_for_check = []
    if chain_cols_to_derive:
        for col in chain_cols_to_derive:
            # MA, STD
            new_features_dict[f"{col}_ma{current_roll_window}"] = data[col].rolling(current_roll_window,
                                                                                    min_periods=max(1,
                                                                                                    current_roll_window // 2)).mean()
            new_features_dict[f"{col}_std{current_roll_window}"] = data[col].rolling(current_roll_window,
                                                                                     min_periods=max(1,
                                                                                                     current_roll_window // 2)).std(
                ddof=0)
            temp_chain_deriv_feats_for_check.extend(
                [f"{col}_ma{current_roll_window}", f"{col}_std{current_roll_window}"])
            # ROC
            new_features_dict[f"{col}_roc1"] = _calculate_refined_roc(data[col], data[col].shift(1))
            temp_chain_deriv_feats_for_check.append(f"{col}_roc1")
            if current_roll_window // 2 > 1:
                new_features_dict[f"{col}_roc{current_roll_window // 2}"] = _calculate_refined_roc(data[col],
                                                                                                   data[col].shift(
                                                                                                       current_roll_window // 2))
                temp_chain_deriv_feats_for_check.append(f"{col}_roc{current_roll_window // 2}")
            # Event-like
            if "whale" in col.lower() or "exchange" in col.lower():
                new_features_dict[f"{col}_active"] = (data[col].abs() > 1e-9).astype(float)
                prev_val = data[col].shift(1)
                new_features_dict[f"{col}_new_event"] = ((prev_val.abs() < 1e-9) & (data[col].abs() > 1e-9)).astype(
                    float)
                temp_chain_deriv_feats_for_check.extend([f"{col}_active", f"{col}_new_event"])
    if temp_chain_deriv_feats_for_check: _check_and_log_feature_issues(
        pd.DataFrame(new_features_dict)[temp_chain_deriv_feats_for_check], temp_chain_deriv_feats_for_check,
        "Chain Derived Features", current_logger)

    # Time Features
    if isinstance(data.index, pd.DatetimeIndex):
        new_features_dict['hour_sin'] = np.sin(2 * np.pi * data.index.hour / 24.0)
        new_features_dict['hour_cos'] = np.cos(2 * np.pi * data.index.hour / 24.0)
        new_features_dict['dayofweek_sin'] = np.sin(2 * np.pi * data.index.dayofweek / 7.0)
        new_features_dict['dayofweek_cos'] = np.cos(2 * np.pi * data.index.dayofweek / 7.0)
        _check_and_log_feature_issues(
            pd.DataFrame(new_features_dict)[['hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos']],
            ['hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos'], "Time Features", current_logger)

    # Concat all new features
    if new_features_dict:
        # Filter out any entries that are not Series (e.g. if a feature calculation failed and returned None)
        valid_new_features = {k: v for k, v in new_features_dict.items() if isinstance(v, pd.Series)}
        if valid_new_features:
            new_features_df = pd.DataFrame(valid_new_features, index=data.index)
            data = pd.concat([data, new_features_df], axis=1)

    # Post-processing from your v3.10
    true_max_lookback = 0  # Calculate based on LAG_STEPS, current_roll_window, atr_window etc.
    # ... (your true_max_lookback calculation) ...
    # Simplified:
    if LAG_STEPS: true_max_lookback = max(LAG_STEPS)
    true_max_lookback = max(true_max_lookback, 26, current_roll_window * 2,
                            atr_window if 'atr_window' in locals() and atr_window else current_roll_window)

    if is_training and len(data) > true_max_lookback:
        data = data.iloc[true_max_lookback:].copy()

    data.replace([np.inf, -np.inf], np.nan, inplace=True)  # Global Inf replace
    current_logger.info("已将所有 Inf 值替换为 NaN.")

    # Fill NaNs (your ffill/bfill logic)
    data.ffill(inplace=True)
    data.bfill(inplace=True)

    # ROC_NAN_FILL_VALUE (your logic, now 0.0)
    for col in data.columns:
        if "_roc" in col.lower() and data[col].isnull().any():  # Check .lower() for safety
            data[col].fillna(ROC_NAN_FILL_VALUE, inplace=True)

    # Chain feature NaN fill (your logic)
    # ... (your logic for filling chain NaNs with 0) ...
    for col_original_chain_name in chain_cols_original_names:
        suffixes_to_check = ["", f"_ma{current_roll_window}", f"_std{current_roll_window}", "_roc1"]
        if current_roll_window // 2 > 1: suffixes_to_check.append(f"_roc{current_roll_window // 2}")
        if "whale" in col_original_chain_name.lower() or "exchange" in col_original_chain_name.lower():
            suffixes_to_check.extend(["_active", "_new_event"])
        for suffix in suffixes_to_check:
            r_col = f"{col_original_chain_name}{suffix}" if suffix else col_original_chain_name
            if r_col in data.columns and data[r_col].isnull().any():

                data[r_col].fillna(0, inplace=True)
                # ------- ATR 回退实现：当 ta 库缺失或前面失败时保证 atr_raw 存在 -------
                if 'atr_raw' not in data.columns:
                    high = data[high_col].astype(float) if high_col in data.columns else None
                    low = data[low_col].astype(float) if low_col in data.columns else None
                    close_prev = data[price_col].shift(1).astype(float) if price_col in data.columns else None
                    if high is not None and low is not None and close_prev is not None:
                        true_range = np.maximum.reduce([
                            (high - low).abs(),
                            (high - close_prev).abs(),
                            (low - close_prev).abs()
                        ])
                        data['atr_raw'] = true_range.rolling(current_roll_window, min_periods=1).mean()
                        current_logger.info(
                            "本机缺 ta.AverageTrueRange，用自带 rolling TrueRange 生成了 atr_raw"
                        )
                # -----------------------------------------------------------------------

    # Remove EXCLUDE_COLS and all-zero columns (your logic)
    if is_training:
        cols_to_remove = [col for col in EXCLUDE_COLS if col in data.columns]
        if cols_to_remove: data.drop(columns=cols_to_remove, errors='ignore', inplace=True)
        # ... (your all-zero column removal logic) ...
        cols_to_keep_if_zero = {price_col, TARGET_COLUMN_NAME, 'atr_raw',
                                'close_raw'}  # Add 'close_raw' if it exists by now
        numeric_cols = data.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            if col not in cols_to_keep_if_zero and (data[col].abs() < 1e-9).all() and not data[
                col].isnull().all():  # Check for all zeros, not all NaNs
                data.drop(columns=[col], inplace=True)
                current_logger.info(f"已删除全零列 {col}")

        # Final NaN drop for features (your logic)
        # Be careful not to drop rows if TARGET_COLUMN_NAME is NaN here
        feature_cols_final_dropna = [c for c in data.columns if c != TARGET_COLUMN_NAME]
        if data[feature_cols_final_dropna].isnull().any().any():
            data.dropna(subset=feature_cols_final_dropna, inplace=True)  # Drop rows if any *feature* is NaN

        # Low variance column removal (your logic)
        # ... (your low variance removal logic, make sure it only considers numeric features) ...
        feature_cols_for_var_check = [c for c in data.select_dtypes(include=np.number).columns if
                                      c not in [TARGET_COLUMN_NAME, "atr_raw", "close_raw"]]
        if feature_cols_for_var_check and len(data) > 1:
            variances = data[feature_cols_for_var_check].var(ddof=0)
            cols_to_drop_low_var = []
            for col_name_v, var_v in variances.items():
                if pd.isna(var_v) or var_v < 1e-9:  # Simplified condition
                    if col_name_v not in {price_col, volume_col, "spread_mean"} and not (
                            "whale" in col_name_v.lower() or "exchange" in col_name_v.lower()):
                        cols_to_drop_low_var.append(col_name_v)
            if cols_to_drop_low_var:
                data.drop(columns=cols_to_drop_low_var, errors='ignore', inplace=True)
                current_logger.info(f"Dropped low/NaN variance columns: {cols_to_drop_low_var}")

    current_logger.info(f"特征工程完成，最终形状 {data.shape}.")
    return data


def preprocess_data(start_dt_str: str, end_dt_str: str, pred_horizon_k_lines: int = PRED_HORIZON_K_CANDLES_DEFAULT,
                    chain_feats: pd.DataFrame = None,
                    agg_period: str = AGGREGATION_PERIOD, roll_window_override: Optional[int] = None,
                    scaler_path_override: Optional[Path] = None, db_engine_override: Optional[Engine] = None) -> Tuple[
    pd.DataFrame, StandardScaler]:
    current_roll_window = roll_window_override if roll_window_override is not None else ROLL_WINDOW
    current_scaler_path_name = f"scaler_{agg_period}_rw{current_roll_window}.joblib"
    current_scaler_path = scaler_path_override if scaler_path_override else (
                Path(ARTIFACTS_DIR) / current_scaler_path_name)
    current_scaler_path.parent.mkdir(parents=True, exist_ok=True)
    logger_a101.info(
        f"Preprocessing: {start_dt_str} to {end_dt_str}. Agg: {agg_period}, RollWin: {current_roll_window}, PredK: {pred_horizon_k_lines}, Scaler: {current_scaler_path}")

    current_db_engine: Engine;
    engine_created_locally = False
    if db_engine_override:
        current_db_engine = db_engine_override; logger_a101.info("使用传入的数据库引擎。")
    else:
        logger_a101.warning(f"未指定数据库引擎，正在创建本地连接 (URL: ...{DATABASE_URL_FALLBACK[-30:]}).")
        try:
            current_db_engine = create_engine(DATABASE_URL_FALLBACK); engine_created_locally = True
        except Exception as e:
            logger_a101.critical(f"Local DB engine creation failed: {e}",
                                 exc_info=True); return pd.DataFrame(), StandardScaler()

    try:
        db_latest_ts = get_db_latest_timestamp(current_db_engine, MARKET_TABLE)
        actual_end_str = end_dt_str
        if db_latest_ts:
            user_end_ts = pd.to_datetime(end_dt_str, utc=True, errors='coerce')
            if user_end_ts and user_end_ts > db_latest_ts: actual_end_str = db_latest_ts.strftime('%Y-%m-%d %H:%M:%S')
        # 1) 行情数据：严格用用户指定的 start → end
        market_df, _ = load_data_from_db(current_db_engine, start_dt_str, actual_end_str)

        # 2) 链上数据：用更长的链上缓冲窗口
        #    这里 MIN_CHAIN_DATA_HISTORY_FOR_A101_STR="30d"（config.py）
        end_ts = pd.to_datetime(actual_end_str, utc=True)
        chain_buffer = pd.Timedelta(MIN_CHAIN_DATA_HISTORY_FOR_A101_STR)
        chain_start_ts = end_ts - chain_buffer
        chain_start_str = chain_start_ts.strftime("%Y-%m-%d %H:%M:%S")

        # ——— 从外部传入的链上特征缓存中切片 ———
        if chain_feats is not None and not chain_feats.empty:
            # cache 里有数据，就直接按本次区间切片
            chain_df = chain_feats.loc[chain_start_str: actual_end_str]
        else:
            # cache 为空或没传入，就从数据库重新加载
            _, chain_df = load_data_from_db(current_db_engine, chain_start_str, actual_end_str)

        chain_cols_orig = chain_df.columns.tolist() if not chain_df.empty else []
        merged = resample_and_merge(market_df, chain_df, agg_period=agg_period)
        if merged.empty or 'close' not in merged.columns: logger_a101.error(
            "❌ 合并后的数据为空或缺少 'close' 列。"); return pd.DataFrame(), StandardScaler()
        labeled = generate_log_return_target(merged, pred_horizon_k=pred_horizon_k_lines)
        if labeled.empty or TARGET_COLUMN_NAME not in labeled.columns: logger_a101.error(
            "❌ 标签生成失败。"); return pd.DataFrame(), StandardScaler()
        featured = add_features(
            labeled,
            chain_cols_orig,
            agg_period=agg_period,
            current_roll_window=current_roll_window,
            is_training=False
        )

        if featured.empty: logger_a101.error("❌ 特征工程失败。"); return pd.DataFrame(), StandardScaler()
        # Essential cols check after add_features
        essential_cols = ['close', TARGET_COLUMN_NAME]
        # atr_raw is created in add_features if TA-Lib is available and data is sufficient
        # if 'atr_raw' in featured.columns : essential_cols.append('atr_raw') # Check if atr_raw was actually created

        for ecol in essential_cols:
            if ecol not in featured.columns:
                logger_a101.error(f"❌ 致命错误：在执行 add_features 后缺少必要列 '{ecol}'！")
                return pd.DataFrame(), StandardScaler()

        normed, scaler = normalize_features(featured, scaler_save_path=current_scaler_path)
        if normed.empty: logger_a101.error("❌ 标准化失败。"); return pd.DataFrame(), scaler
        if TARGET_COLUMN_NAME in normed.columns: normed.dropna(subset=[TARGET_COLUMN_NAME], inplace=True)
        if isinstance(normed.index, pd.DatetimeIndex): normed.index.name = "time_idx"
        logger_a101.info(f"预处理完成。最终形状：{normed.shape}。.")
        return normed, scaler
    finally:
        if engine_created_locally and 'current_db_engine' in locals() and current_db_engine:
            current_db_engine.dispose();
            logger_a101.info("已释放本地数据库连接。")



if __name__ == "__main__":
    if not logger_a101.hasHandlers(): logging.basicConfig(level=logging.INFO,
                                                          format="%(asctime)s [%(name)s] %(levelname)s - %(message)s")
    # ... (rest of your __main__ demo logic, it should work with the above functions) ...
    # Make sure ARTIFACTS_DIR is correctly defined or imported for the demo scaler path
    demo_agg_period = AGGREGATION_PERIOD
    demo_roll_window = ROLL_WINDOW
    demo_pred_horizon = PRED_HORIZON_K_CANDLES_DEFAULT
    demo_scaler_path_name = f"scaler_demo_{demo_agg_period}_rw{demo_roll_window}.joblib"
    demo_scaler_path = Path(ARTIFACTS_DIR) / demo_scaler_path_name  # Use ARTIFACTS_DIR

    demo_start, demo_end = "2025-01-20 00:00:00", "2025-01-22 00:00:00"
    logger_a101.info(
        f"演示模式开始：{demo_start} → {demo_end}；聚合周期：{demo_agg_period}，滚动窗口：{demo_roll_window}，预测跨度：{demo_pred_horizon}")
    if demo_scaler_path.exists(): demo_scaler_path.unlink()

    df_ready, loaded_scaler = preprocess_data(
        demo_start, demo_end,
        pred_horizon_k_lines=demo_pred_horizon, agg_period=demo_agg_period,
        roll_window_override=demo_roll_window, scaler_path_override=demo_scaler_path,
        db_engine_override=None
    )
    if not df_ready.empty:
        logger_a101.info(f"✦ DEMO: Final shape: {df_ready.shape}")
    else:
        logger_a101.error("✦ DEMO: Preprocessing failed.")
    logger_a101.info("DEMO run finished.")