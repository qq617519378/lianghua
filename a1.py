#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模块1：数据预处理与融合（混合版：保留原始与新增衍生特征，使用 blockchain_features_utc，无需时区转换）
----------------------------------------------
功能：
1. 加载 crypto_data（行情）与 blockchain_features_utc（链上特征）。
2. 行情按指定频率重采样。
3. 链上特征解析为 UTC，无需本地时区转换。
4. 合并两表数据，并填充空值。
5. 构建【原始衍生特征】+【新增高价值特征】。
6. 标准化所有数值型特征，自动检测 scaler 是否匹配，否则重新 fit 并保存。
7. 返回处理后 DataFrame，并在示例中完整打印几列前 5 行，供验证。

依赖： pandas, numpy, sqlalchemy, scikit-learn, logging, joblib
"""
import config

import pandas as pd
import numpy as np
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sklearn.preprocessing import StandardScaler
import joblib
import os
from sqlalchemy import create_engine, Engine # 确保导入 Engine
from typing import Optional
from config import MERGE_TOLERANCE_MINUTES
# Pandas 完整显示设置
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# 重采样规则
CRYPTO_AGG_RULES = {
    'open_ask_price': 'first', 'high_ask_price': 'max', 'low_ask_price': 'min', 'close_ask_price': 'last',
    'open_bid_price': 'first', 'high_bid_price': 'max', 'low_bid_price': 'min', 'close_bid_price': 'last',
    'spread_min': 'mean', 'spread_max': 'mean', 'spread_mean': 'mean', 'spread_twa': 'mean',
    'ask_depth_min': 'mean', 'ask_depth_max': 'mean', 'ask_depth_mean': 'mean', 'ask_depth_twa': 'mean',
    'bid_depth_min': 'mean', 'bid_depth_max': 'mean', 'bid_depth_mean': 'mean', 'bid_depth_twa': 'mean',
    'imbalance_min': 'mean', 'imbalance_max': 'mean', 'imbalance_mean': 'mean', 'imbalance_twa': 'mean',
    'kline_open': 'first', 'kline_high': 'max', 'kline_low': 'min', 'kline_close': 'last', 'kline_volume': 'sum',
'spread_raw':    'mean',
    'mid_price_raw': 'mean',
    'micro_price_raw': 'mean'
}


# 链上特征列（排除 id、height、mempool_size、total_mempool_value、avg_mempool_fee_rate）
BC_COLUMNS = [
    'block_size', 'stripped_size', 'weight', 'tx_count', 'input_count', 'output_count',
    'segwit_tx_ratio', 'taproot_tx_ratio', 'block_interval', 'difficulty', 'block_reward',
    'fee_to_reward_ratio', 'size_utilization', 'mean_fee_rate', 'median_fee_rate', 'p90_fee_rate',
    'avg_tx_size', 'median_tx_size', 'large_tx_count', 'opreturn_tx_count', 'opreturn_total_bytes',
    'active_addresses', 'new_addresses', 'total_fee', 'fee_min', 'fee_max', 'fee_median', 'fee_mean',
    'fee_p25', 'fee_p75', 'fee_count', 'exchange_inflow', 'exchange_outflow', 'exchange_netflow',
    'whale_to_exchange', 'whale_from_exchange', 'whale_non_exchange'
]

class NewDataProcessor:
    def __init__(self,
                 conn_str: Optional[str] = None,  # 改为 Optional，因为可能通过引擎传入
                 freq: str = '30s',
                 db_engine_override: Optional[Engine] = None  # <--- 新增参数
                 ):

        if db_engine_override is not None:
            self.engine = db_engine_override
            logging.info(f"NewDataProcessor（a1）使用传入的数据库引擎。频率：{freq}")

        elif conn_str is not None:
            self.engine = create_engine(conn_str)
            logging.info(f"NewDataProcessor（a1）使用 conn_str 创建了数据库引擎。频率：{freq}")

        else:
            # 如果两者都没有，则抛出错误，因为该类依赖于引擎
            logging.error("❌ NewDataProcessor (a1) 需要提供 db_engine_override 或 conn_str。")
            raise ValueError("❌ NewDataProcessor (a1) 必须使用数据库引擎或连接字符串进行初始化。")

        self.freq = freq
        self.scaler_path = f'scaler_{self.freq}.joblib'  # scaler_path 的定义移到引擎确定之后
        if os.path.exists(self.scaler_path):
            try:
                self.scaler = joblib.load(self.scaler_path)
                logging.info(f'加载已有 scaler (a1): {self.scaler_path}')
            except Exception:
                self.scaler = None
                logging.error('❌ 运行时未找到训练期 scaler (a1)，程序终止。')
                raise FileNotFoundError('❌ 未找到 A1 训练期 scaler 文件：scaler_a1.joblib')


    def load_crypto_data(self, start: str, end: str) -> pd.DataFrame:
        sql = '''SELECT * FROM crypto_data WHERE time_window >= :start AND time_window < :end'''
        try:
            df = pd.read_sql(text(sql), self.engine, params={'start': start, 'end': end})
            df['time_window'] = pd.to_datetime(df['time_window'], utc=True)
            df = df.sort_values('time_window').drop_duplicates('time_window')
            # 高频价差与 micro-price
            df['mid_price_raw'] = (df['close_ask_price'] + df['close_bid_price']) / 2
            df['spread_raw'] = (df['close_ask_price'] - df['close_bid_price']).abs()
            depth_sum = (df['ask_depth_mean'] + df['bid_depth_mean']).replace(0, np.nan)
            df['micro_price_raw'] = (
                                            df['close_ask_price'] * df['bid_depth_mean'] +
                                            df['close_bid_price'] * df['ask_depth_mean']
                                    ) / depth_sum
            df['micro_price_raw'] = df['micro_price_raw'].fillna(df['mid_price_raw'])
            logging.info(f'加载行情数据: {len(df)} 条')
            return df
        except SQLAlchemyError as e:
            logging.error(f'加载行情数据失败: {e}')
            return pd.DataFrame()

    def load_blockchain(self, start: str, end: str) -> pd.DataFrame:
        sql = '''SELECT * FROM blockchain_features_utc WHERE timestamp >= :start AND timestamp < :end'''
        try:
            df = pd.read_sql(text(sql), self.engine, params={'start': start, 'end': end})
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df = df.sort_values('timestamp').drop_duplicates('timestamp')
            logging.info(f'加载链上特征: {len(df)} 条')
            return df
        except SQLAlchemyError as e:
            logging.error(f'加载链上特征失败: {e}')
            return pd.DataFrame()

    def resample_crypto(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        # ——— 如果缺少合成列，就先算一下 ———
        if 'mid_price_raw' not in df.columns:
            # 中间价 = (卖一价 + 买一价)/2
            df['mid_price_raw'] = (df['close_ask_price'] + df['close_bid_price']) / 2
            # 原始价差
            df['spread_raw'] = (df['close_ask_price'] - df['close_bid_price']).abs()
            # 微价：根据挂单深度加权
            depth_sum = (df['ask_depth_mean'] + df['bid_depth_mean']).replace(0, np.nan)
            df['micro_price_raw'] = (
                                            df['close_ask_price'] * df['bid_depth_mean']
                                            + df['close_bid_price'] * df['ask_depth_mean']
                                    ) / depth_sum
            # 深度为 0 时退回到中间价
            df['micro_price_raw'].fillna(df['mid_price_raw'], inplace=True)

        # —— 新增：保证有 time_window 列 —— #
        if 'time_window' not in df.columns:
            # df.index.name 可能是 'timestamp' 或者 None
            idx_name = df.index.name or 'index'
            df = df.reset_index().rename(columns={idx_name: 'time_window'})

        # 原有逻辑：按 time_window 重采样
        df_res = (
            df
            .set_index('time_window')
            .resample(self.freq)
            .agg(CRYPTO_AGG_RULES)
            .reset_index()
        )
        logging.info(f'行情重采样后: {len(df_res)} 条')
        return df_res

    def merge_data(
            self,
            cd: pd.DataFrame,
            bf: pd.DataFrame,
            tolerance: str | pd.Timedelta = "120min"  # ⇐ 想放宽就把默认值改成 "60min"
    ) -> pd.DataFrame:
        """
        将重采样后的行情 (cd) 与链上特征 (bf) 对齐合并
        - 先把 bf 的索引补回 timestamp 列
        - 只前向填充，不用 0 填充
        - tolerance 参数决定最大对齐间隔
        """
        # 1) 基本检查
        if cd.empty or bf.empty:
            return pd.DataFrame()

        # 2) 保证 bf 以 DatetimeIndex 为索引，并补一列 timestamp 供后续使用
        if not isinstance(bf.index, pd.DatetimeIndex):
            if "timestamp" in bf.columns:
                bf["timestamp"] = pd.to_datetime(bf["timestamp"], utc=True, errors="coerce")
                bf.set_index("timestamp", inplace=True)
            else:
                raise ValueError("链上 DataFrame 缺少可用的时间索引/列")

        bf = bf.copy()
        bf["timestamp"] = bf.index  # 补回列

        # 3) 对齐索引并前向填充
        idx = pd.date_range(
            cd["time_window"].min(),
            cd["time_window"].max(),
            freq=self.freq,
            tz="UTC"
        )
        bf_reindexed = bf.reindex(idx).ffill()

        # 4) merge_asof 按 tolerance 对齐
        merged = pd.merge_asof(
            cd.sort_values("time_window"),
            bf_reindexed[BC_COLUMNS].sort_index(),
            left_on="time_window",
            right_index=True,
            direction="backward",
            tolerance=pd.Timedelta(tolerance)
        )

        # 5) 去掉在 BC_COLUMNS 里仍全缺失的行，再返回
        merged.dropna(subset=BC_COLUMNS, inplace=True)
        return merged.fillna(0)  # 这里只填剩下的 NaN，特征本身已通过缓存计算出来

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        eps = 1e-6; N = 10
        # 原始衍生特征
        df['price_range'] = df['kline_high'] - df['kline_low']
        df['close_volatility'] = df['kline_close'].rolling(30, min_periods=1).std().fillna(0)
        df['price_momentum'] = df['kline_close'].diff().fillna(0)
        df['bid_ask_diff'] = df['open_ask_price'] - df['open_bid_price']
        df['active_tx_ratio'] = (df['active_addresses'] / df['tx_count'].replace(0, np.nan)).fillna(0)
        df['fee_spread'] = df['fee_max'] - df['fee_min']
        df['fee_diff'] = df['fee_mean'] - df['fee_median']
        df['fee_ratio_calc'] = (df['fee_max'] / df['fee_min'].replace(0, np.nan)).fillna(0)
        # 新增高价值特征
        df['block_interval_rate'] = df['block_interval'].pct_change().fillna(0)
        df['difficulty_rate'] = df['difficulty'].pct_change().fillna(0)
        df['tx_count_mom'] = df['tx_count'].diff().fillna(0)
        df['tx_rate'] = (df['tx_count'] / (df['block_interval'] + eps)).fillna(0)
        df['io_ratio'] = (df['input_count'] / (df['output_count'] + eps)).fillna(0)
        df['segwit_ratio_change'] = df['segwit_tx_ratio'].diff().fillna(0)
        df['taproot_ratio_change'] = df['taproot_tx_ratio'].diff().fillna(0)
        df['fee_p90_spread'] = (df['p90_fee_rate'] - df['median_fee_rate']).fillna(0)
        df['size_util_std'] = df['size_utilization'].rolling(N, min_periods=1).std().fillna(0)
        df['fee_rate_std'] = df['mean_fee_rate'].rolling(N, min_periods=1).std().fillna(0)
        df['interval_std'] = df['block_interval'].rolling(N, min_periods=1).std().fillna(0)
        df['large_tx_ratio'] = (df['large_tx_count'].rolling(N).sum() / df['tx_count'].rolling(N).sum()).fillna(0)
        df['opreturn_tx_ratio'] = (df['opreturn_tx_count'].rolling(N).sum() / df['tx_count'].rolling(N).sum()).fillna(0)
        df['reward_fee_ratio'] = (df['total_fee'] / (df['block_reward'] + eps)).fillna(0)
        df['size_fee_ratio'] = (df['size_utilization'] / (df['fee_mean'] + eps)).fillna(0)
        df['whale_net'] = (df['whale_to_exchange'] - df['whale_from_exchange']).fillna(0)
        df['whale_to_exchange_ratio'] = (df['whale_to_exchange'] / (df['exchange_inflow'] + eps)).fillna(0)
        df['whale_activity'] = ((df['whale_to_exchange'] > 0) | (df['whale_from_exchange'] > 0)).astype(int)
        df['whale_activity_count'] = df['whale_activity'].rolling(N, min_periods=1).sum().astype(int)
        # ===== 新增订单簿 / 价格型技术指标 =====
        df['mid_return'] = df['mid_price_raw'].pct_change().fillna(0)
        df['micro_return'] = df['micro_price_raw'].pct_change().fillna(0)

        for w in (2, 5, 10):
            df[f'atr_{w}'] = (df['kline_high'] - df['kline_low']).rolling(w).mean().fillna(0)
            df[f'vol_std_{w}'] = df['kline_volume'].rolling(w).std().fillna(0)
            df[f'price_z_{w}'] = (
                    (df['micro_price_raw'] - df['micro_price_raw'].rolling(w).mean()) /
                    (df['micro_price_raw'].rolling(w).std() + eps)
            ).fillna(0)

        # EMA & RSI
        for span in (3, 6, 12):
            df[f'ema_{span}'] = df['micro_price_raw'].ewm(span=span, adjust=False).mean()

        delta = df['micro_price_raw'].diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / (loss + eps)
        df['rsi_14'] = 100 - 100 / (1 + rs)

        # 深度方向性指标
        df['depth_imbalance'] = (
                (df['bid_depth_mean'] - df['ask_depth_mean']) /
                (df['bid_depth_mean'] + df['ask_depth_mean'] + eps)
        ).fillna(0)
        df['liquidity_pressure'] = df['spread_raw'] / (df['depth_imbalance'].abs() + eps)

        # 成交量暴涨 flag
        vol_mean = df['kline_volume'].rolling(5).mean()
        vol_std = df['kline_volume'].rolling(5).std()
        df['volume_spike'] = (df['kline_volume'] > vol_mean + 2 * vol_std).astype(int)
        df['dprice'] = df['kline_close'].astype(float).diff().fillna(0.0)

        logging.info('衍生特征计算完成')
        return df

    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用训练期生成的 scaler 对行情特征进行标准化。
        若列数或顺序与 scaler 不一致，立即抛错终止，绝不重新拟合。
        """
        if df.empty:
            logging.warning("normalize_data: 传入 DataFrame 为空")
            return df

        # 1. 识别数值列（排除原价与点差原始值）
        numeric_cols = (
            df.select_dtypes(include=[np.number]).columns
            .difference(["mid_price_raw", "micro_price_raw", "spread_raw"])
        )
        # PATCH NaN fallback
        if df.loc[:, numeric_cols].isna().all(axis=None):
            logging.warning("normalize_data: 全列 NaN，返回原始 DataFrame 以便上层 ffill")
            return df

        # 2. 处理无穷值
        df.loc[:, numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

        # 3. 缺失值填充
        df.loc[:, numeric_cols] = (
            df[numeric_cols]
            .ffill()
            .bfill()
            .fillna(0.0)
        )

        # 4. 加载并验证 scaler
        if self.scaler is None:
            if not self.scaler_path.exists():
                logging.error(f"A1_Scaler_Missing: 未找到训练期 scaler 文件 {self.scaler_path}")
                raise FileNotFoundError(f"Scaler not found: {self.scaler_path}")
            try:
                self.scaler: StandardScaler = joblib.load(self.scaler_path)
            except Exception as e:
                logging.error(f"A1_Scaler_Load_Failed: {e}")
                raise RuntimeError("A1_Scaler_Load_Failed") from e

        # 5. 检查特征维度与顺序
        saved_order = list(getattr(self.scaler, "feature_names_in_", []))
        if len(saved_order) != len(numeric_cols) or set(saved_order) != set(numeric_cols):
            logging.error("A1_Scaler_Incompatible: 当前特征列与训练期 scaler 不一致")
            raise RuntimeError("A1_Scaler_Incompatible")

        # 6. 按训练期顺序 transform
        df.loc[:, saved_order] = self.scaler.transform(df[saved_order].astype(float))
        logging.info("标准化完成（使用训练期 scaler）")

        return df

    def get_processed_data(self, start: str, end: str) -> pd.DataFrame:
        cd = self.load_crypto_data(start, end)
        bf = self.load_blockchain(start, end)
        rs = self.resample_crypto(cd)
        merged = self.merge_data(rs, bf)
        feat = self._add_technical_indicators(merged)
        norm = self.normalize_data(feat)
        norm.dropna(inplace=True)
        norm.set_index('time_window', inplace=True, drop=False)
        norm.index.rename('time_idx', inplace=True)
        norm.sort_index(inplace=True)
        return norm

    def get_processed_data_from_df(self, crypto_df: pd.DataFrame, chain_df: pd.DataFrame) -> pd.DataFrame:
        """
        接收内存中已有的行情和链上特征 DataFrame，
        按照 get_processed_data 一致的流程执行：重采样、合并、特征工程、标准化、索引重置。
        """
        # 重采样行情
        rs = self.resample_crypto(crypto_df)
        # 合并链上特征
        merged = self.merge_data(rs, chain_df)
        # 添加技术指标
        feat = self._add_technical_indicators(merged)
        # 标准化
        norm = self.normalize_data(feat)
        # 丢弃 NaN
        norm.dropna(inplace=True)
        # 设 time_window 为索引
        norm.set_index('time_window', inplace=True, drop=False)
        norm.index.rename('time_idx', inplace=True)
        norm.sort_index(inplace=True)
        return norm


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    proc = NewDataProcessor('postgresql+psycopg2://postgres:456258@localhost:5432/crypto_data', freq='30s')
    df = proc.get_processed_data('2025-01-20 00:00:00', '2025-01-22 00:00:00')
    if not df.empty:
        print('示例衍生特征前5行:')
        cols_to_show = [
            'price_range', 'close_volatility', 'price_momentum', 'bid_ask_diff',
            'active_tx_ratio', 'fee_spread', 'fee_diff', 'fee_ratio_calc', 'whale_net'
        ]
        print(df[cols_to_show].head(5).to_string(index=False))
    else:
        print('处理结果为空')
