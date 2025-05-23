#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
a201_price_regression_v1.4.2.py - 确保检查点保存所有必要元数据
================================================================================
(改动 v1.4.2)
- 重点审查并确认 main_train_pytorch_regression 中保存检查点时，
  包含了模型配置、特征列表、数据处理配置等所有关键元数据。
  (实际上原代码已经做得很好，此处主要是确认和细微调整)

(继承自 v1.4.1)
- 将脚本中的主要英文日志信息翻译为中文。

(继承自 v1.4.0)
- 实现 Within-Bucket MixUp (基于atr_raw分桶)。
- 实现自动特征文件检测与回退。
- 移除 --use_lgbm_features 参数。

(继承自 v1.3.0)
- 多任务学习 (Log Return + ΔPrice)。
- 复合指标 Early-Stopping。
- 优化验证阶段 EMA 应用、DataLoader pin_memory、Cosine T_0、验证混合精度。
- ...
"""
from collections import Counter
import logging, random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union, Optional, Iterator
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.amp import autocast, GradScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import config

try:
    import a101
except ImportError:
    logging.error("无法导入 a101.py 模块。请确保它在正确的 Python 路径下且依赖已安装。")
    raise

MAX_DPRICE_CLIP = 200.0


def dprice_encode(raw: torch.Tensor) -> torch.Tensor:
    raw = torch.clamp(raw, -MAX_DPRICE_CLIP, MAX_DPRICE_CLIP)
    return torch.tanh(raw / MAX_DPRICE_CLIP)


def dprice_decode(encoded: torch.Tensor) -> torch.Tensor:
    return torch.arctanh(encoded) * MAX_DPRICE_CLIP


def focal_mse(pred, target, gamma: float = 2.0):
    err = pred - target
    pt = torch.exp(-err ** 2)
    return ((1 - pt) ** gamma * err ** 2).mean()


smooth_l1 = torch.nn.SmoothL1Loss(beta=0.02, reduction='mean')

SEQ_LEN = 20  # a201.py 自己的 SEQ_LEN，与 trading_orchestrator.py 中的 A201_MODULE_SEQ_LEN 对应
LGBM_FEATURE_FILE_DEFAULT = "lgbm_top_features_for_pytorch.txt"
N_VOLATILITY_BUCKETS = 4

BATCH_SIZE_DEFAULT = 1024
LR_BASE_DEFAULT = 1e-4
WARMUP_EPOCHS_DEFAULT = 10
PATIENCE_DEFAULT = 25
EPOCHS_DEFAULT = 300
D_MODEL_DEFAULT = 128
GRU_HIDDEN_DEFAULT = 64
N_BLOCK_DEFAULT = 5
GRU_LAYERS_DEFAULT = 2
N_LAYER_DEFAULT = 2
WEIGHT_DECAY_DEFAULT = 1e-5
DROPOUT_TRANSFORMER_LAYER_DEFAULT = 0.1
DROPOUT_MODEL_HEAD_DEFAULT = 0.2
EMA_DECAY = 0.995
SWA_START_FRAC_DEFAULT = 0.25
LOSS_WEIGHT_LOGRET = 1.0
LOSS_WEIGHT_DELTA_PRICE = 0.3

START_DATE_DEFAULT = "2025-01-20 00:00:00"
END_DATE_DEFAULT = "2027-01-01 00:00:00"

A101_TARGET_COL = getattr(a101, 'TARGET_COLUMN_NAME', 'future_log_return')
A101_PRED_HORIZON_K = getattr(a101, 'PRED_HORIZON_K_CANDLES_DEFAULT', 1)
A101_AGG_PERIOD = getattr(a101, 'AGGREGATION_PERIOD', "3min")
A101_ROLL_WINDOW = getattr(a101, 'ROLL_WINDOW', 15)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt='%Y-%m-%d %H:%M:%S')


class PriceRegressionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_names: List[str], target_name: str, seq_len: int = SEQ_LEN,
                 n_buckets: int = N_VOLATILITY_BUCKETS):
        required_cols = [target_name, 'close_raw', 'atr_raw'] + feature_names
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"数据集 DataFrame 缺少必要的列: {missing_cols}")

        self.x_features = df[feature_names].values.astype(np.float32)
        self.y_target_logret = df[target_name].values.astype(np.float32)
        self.close_raw = df['close_raw'].values.astype(np.float32)
        self.atr_raw = df['atr_raw'].values.astype(np.float32)

        self.seq_len = seq_len
        self.num_total_rows = len(df)
        self.num_samples = self.num_total_rows - seq_len + 1

        if self.num_samples <= 0:
            logging.error(f"DataFrame 长度 ({self.num_total_rows}) 不足以创建长度为 {seq_len} 的序列。")
            raise ValueError("数据长度不足以创建序列。")
        if self.x_features.shape[1] == 0:
            logging.error("数据集 X 中没有特征列。")
            raise ValueError("数据集 X 没有特征。")

        self.n_buckets = n_buckets
        atr_at_seq_end = self.atr_raw[self.seq_len - 1:]
        assert len(atr_at_seq_end) == self.num_samples, "ATR 分桶长度不匹配"

        if self.n_buckets > 1 and len(np.unique(atr_at_seq_end[~np.isnan(atr_at_seq_end)])) >= self.n_buckets:
            try:
                quantiles = np.linspace(0, 1, self.n_buckets + 1)[1:-1]
                bucket_edges = np.quantile(atr_at_seq_end[~np.isnan(atr_at_seq_end)], quantiles)
                self.bucket_id = np.digitize(atr_at_seq_end, bucket_edges, right=False).astype(np.int8)
                logging.info(f"数据集: 基于 'atr_raw' 的分位数创建了 {self.n_buckets} 个波动率桶。边界: {bucket_edges}")
                bucket_counts = Counter(self.bucket_id)
                logging.info(f"数据集: 桶内样本分布: {dict(sorted(bucket_counts.items()))}")
            except Exception as e:
                logging.warning(f"数据集: 使用 atr_raw 创建波动率桶失败: {e}。将使用单个桶 (ID 0)。")
                self.bucket_id = np.zeros(self.num_samples, dtype=np.int8)
        else:
            logging.warning(f"数据集: 唯一波动率值不足或 n_buckets <= 1。将使用单个桶 (ID 0)。")
            self.bucket_id = np.zeros(self.num_samples, dtype=np.int8)

        logging.debug(
            f"数据集创建完毕。样本数: {self.num_samples}, 特征数: {self.x_features.shape[1]}, 桶数: {len(np.unique(self.bucket_id))}")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start_idx = idx
        end_idx = idx + self.seq_len
        x_sequence = torch.from_numpy(self.x_features[start_idx:end_idx])
        label_logret = torch.tensor(self.y_target_logret[end_idx - 1]).unsqueeze(0)
        current_close_raw = torch.tensor(self.close_raw[end_idx - 1]).unsqueeze(0)
        return x_sequence, label_logret, current_close_raw


class BucketBatchSampler(Sampler[List[int]]):
    def __init__(self, bucket_ids: np.ndarray, batch_size: int, shuffle: bool = True, drop_last: bool = False):
        if not isinstance(bucket_ids, np.ndarray):
            bucket_ids = np.array(bucket_ids)
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正数")

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.bucket_ids = bucket_ids
        self.buckets: Dict[int, List[int]] = {}
        unique_buckets = np.unique(bucket_ids)
        for bucket_id in unique_buckets:
            self.buckets[int(bucket_id)] = np.where(bucket_ids == bucket_id)[0].tolist()

        self.num_batches_per_bucket = {}
        self.total_batches = 0
        for bucket_id, indices in self.buckets.items():
            num_batches = len(indices) // self.batch_size
            if not self.drop_last and len(indices) % self.batch_size > 0:
                num_batches += 1
            self.num_batches_per_bucket[bucket_id] = num_batches
            self.total_batches += num_batches
        logging.info(
            f"桶批次采样器 (BucketBatchSampler) 初始化完毕。桶数: {len(self.buckets)}, 总批次数: {self.total_batches}, 批大小: {self.batch_size}, 丢弃末尾: {self.drop_last}")

    def __iter__(self) -> Iterator[List[int]]:
        all_batches: List[List[int]] = []
        bucket_keys = list(self.buckets.keys())
        if self.shuffle:
            random.shuffle(bucket_keys)
            for bucket_id in bucket_keys:
                random.shuffle(self.buckets[bucket_id])
        for bucket_id in bucket_keys:
            indices = self.buckets[bucket_id]
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i: i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                all_batches.append(batch)
        if self.shuffle:
            random.shuffle(all_batches)
        yield from all_batches

    def __len__(self) -> int:
        return self.total_batches


class TimesBlock(nn.Module):
    def __init__(self, d_model: int, kernel_set: List[int] = [1, 3, 5, 7]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, k, padding=k // 2, groups=d_model) for k in kernel_set])
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.transpose(1, 2)
        y = sum(conv(z) for conv in self.convs)
        y = self.act(self.bn(y)).transpose(1, 2)
        return x + y


class AttnPooling(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        self.query_param = nn.Parameter(torch.randn(1, 1, d_model))
        self.attn = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        queries = self.query_param.repeat(x.size(0), 1, 1)
        out, _ = self.attn(queries, x, x, need_weights=False)
        return out.squeeze(1)


class PricePredictorModel(nn.Module):
    def __init__(self,
                 in_dim: int,
                 d_model: int,
                 gru_hidden: int,
                 n_block: int,
                 gru_layers: int,
                 n_layer: int,
                 dropout_transformer: float,
                 dropout_head: float,
                 nhead_transformer: int = 8):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.tblocks = nn.ModuleList([TimesBlock(d_model) for _ in range(n_block)])
        self.bigru = nn.GRU(d_model, gru_hidden, num_layers=gru_layers, batch_first=True, bidirectional=True)
        fusion_dim = d_model + (2 * gru_hidden)
        assert fusion_dim % nhead_transformer == 0, \
            f"Transformer 的 nhead ({nhead_transformer}) 必须能整除融合维度 fusion_dim ({fusion_dim})!"
        encoder_layer_config = nn.TransformerEncoderLayer(
            d_model=fusion_dim, nhead=nhead_transformer, dim_feedforward=fusion_dim * 2,
            batch_first=True, dropout=dropout_transformer)
        self.encoder = nn.TransformerEncoder(encoder_layer_config, num_layers=n_layer)
        self.pool = AttnPooling(fusion_dim, num_heads=4)
        self.head_norm = nn.LayerNorm(fusion_dim)
        self.head_shared_linear = nn.Linear(fusion_dim, fusion_dim // 2)
        self.head_act = nn.GELU()
        self.head_dropout = nn.Dropout(dropout_head)
        self.logret_head = nn.Linear(fusion_dim // 2, 1)
        self.price_delta_head = nn.Linear(fusion_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.bigru.flatten_parameters()
        h = self.proj(x)
        h_timesnet = h
        for blk in self.tblocks: h_timesnet = blk(h_timesnet)
        h_gru, _ = self.bigru(h)
        h_cat = torch.cat([h_timesnet, h_gru], dim=-1)
        h_enc = self.encoder(h_cat)
        pooled_output = self.pool(h_enc)
        h_head = self.head_norm(pooled_output)
        h_head = self.head_shared_linear(h_head)
        h_head = self.head_act(h_head)
        h_head = self.head_dropout(h_head)
        log_ret_output = self.logret_head(h_head)
        price_delta_output = self.price_delta_head(h_head)
        return log_ret_output, price_delta_output


class EMA:  # Duplicated EMA class, using the one at the top of trading_orchestrator.py is better.
    # For this file, it's fine as it's self-contained for training.
    def __init__(self, model: nn.Module, decay: float = EMA_DECAY):
        self.decay = decay;
        self.shadow_params = {name: param.detach().clone().float()
                              for name, param in model.state_dict().items() if param.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.state_dict().items():
            if name in self.shadow_params and param.dtype.is_floating_point:
                self.shadow_params[name].mul_(self.decay).add_(param.float(), alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module):
        model_state_dict = model.state_dict()
        for name, shadow_val in self.shadow_params.items():
            if name in model_state_dict: model_state_dict[name].copy_(shadow_val)


@torch.no_grad()
def evaluate_log_return_regression(
        model: nn.Module,
        loader: DataLoader,
        criterion_logret: nn.Module,
        criterion_price_delta: nn.Module,
        loss_weight_logret: float,
        loss_weight_price_delta: float
):
    model.eval()
    if loader is None:
        return (0.0,) * 8 + (torch.empty(0),) * 5
    all_true_lr, all_pred_lr, all_true_dp, all_pred_dp, all_close_raw = [], [], [], [], []
    tot_loss, n_samples = 0.0, 0
    for x, y_lr, close_raw in loader:
        x, y_lr, close_raw = x.to(DEVICE), y_lr.to(DEVICE), close_raw.to(DEVICE)
        with autocast(device_type=DEVICE.type, enabled=(DEVICE.type == 'cuda')):
            pred_lr, pred_dp_raw = model(x)
            pred_dp = dprice_encode(pred_dp_raw)
        tgt_dp_raw = close_raw * (torch.exp(y_lr) - 1.0)
        tgt_dp = dprice_encode(tgt_dp_raw)
        loss_lr = criterion_logret(pred_lr, y_lr)
        loss_dp = criterion_price_delta(pred_dp, tgt_dp)
        loss = loss_weight_logret * loss_lr + loss_weight_price_delta * loss_dp
        tot_loss += loss.item() * y_lr.size(0)
        n_samples += y_lr.size(0)
        all_true_lr.append(y_lr.cpu());
        all_pred_lr.append(pred_lr.cpu())
        all_true_dp.append(tgt_dp.cpu());
        all_pred_dp.append(pred_dp.cpu())
        all_close_raw.append(close_raw.cpu())
    if n_samples == 0:
        return (0.0,) * 8 + (torch.empty(0),) * 5
    true_lr = torch.cat(all_true_lr).squeeze();
    pred_lr = torch.cat(all_pred_lr).squeeze()
    true_dp = torch.cat(all_true_dp).squeeze();
    pred_dp = torch.cat(all_pred_dp).squeeze()
    close_px = torch.cat(all_close_raw).squeeze()
    true_dp_dec = dprice_decode(true_dp);
    pred_dp_dec = dprice_decode(pred_dp)
    avg_comb_loss = tot_loss / n_samples
    mae_lr = mean_absolute_error(true_lr.numpy(), pred_lr.numpy())
    rmse_lr = np.sqrt(mean_squared_error(true_lr.numpy(), pred_lr.numpy()))
    mae_dp = mean_absolute_error(true_dp_dec.numpy(), pred_dp_dec.numpy())
    rmse_dp = np.sqrt(mean_squared_error(true_dp_dec.numpy(), pred_dp_dec.numpy()))
    true_abs_future = close_px * torch.exp(true_lr);
    pred_abs_future = close_px * torch.exp(pred_lr)
    mae_px = mean_absolute_error(true_abs_future.numpy(), pred_abs_future.numpy())
    rmse_px = np.sqrt(mean_squared_error(true_abs_future.numpy(), pred_abs_future.numpy()))
    composite = 0.7 * rmse_px + 0.3 * rmse_lr
    return (avg_comb_loss, composite, mae_lr, rmse_lr, mae_dp, rmse_dp, mae_px, rmse_px,
            true_lr, pred_lr, true_dp_dec, pred_dp_dec, close_px)


def mixup_data_regression(
        x: torch.Tensor,
        y_target_logret: torch.Tensor,
        current_close_raw: torch.Tensor,
        alpha: float = 0.2
) -> Tuple[
    torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]]:
    if alpha <= 0:
        target_delta_price_raw = current_close_raw * (torch.exp(y_target_logret) - 1.0)
        target_delta_price = dprice_encode(target_delta_price_raw)
        return x, (y_target_logret, y_target_logret, target_delta_price, target_delta_price,
                   current_close_raw, current_close_raw, 1.0)
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    permuted_indices = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[permuted_indices, :]
    y_a_logret, y_b_logret = y_target_logret, y_target_logret[permuted_indices]
    close_raw_a, close_raw_b = current_close_raw, current_close_raw[permuted_indices]
    y_a_delta_raw = close_raw_a * (torch.exp(y_a_logret) - 1.0)
    y_b_delta_raw = close_raw_b * (torch.exp(y_b_logret) - 1.0)
    y_a_delta = dprice_encode(y_a_delta_raw);
    y_b_delta = dprice_encode(y_b_delta_raw)
    return mixed_x, (y_a_logret, y_b_logret, y_a_delta, y_b_delta, close_raw_a, close_raw_b, lam)


def mixup_criterion_regression(
        criterion_logret: nn.Module, criterion_price_delta: nn.Module,
        pred_logret: torch.Tensor, pred_price_delta: torch.Tensor,
        y_a_logret: torch.Tensor, y_b_logret: torch.Tensor,
        y_a_delta: torch.Tensor, y_b_delta: torch.Tensor,
        lam: float,
        loss_weight_logret: float, loss_weight_price_delta: float
) -> torch.Tensor:
    loss_logret_mix = lam * criterion_logret(pred_logret, y_a_logret) + \
                      (1 - lam) * criterion_logret(pred_logret, y_b_logret)
    loss_price_delta_mix = lam * criterion_price_delta(pred_price_delta, y_a_delta) + \
                           (1 - lam) * criterion_price_delta(pred_price_delta, y_b_delta)
    return loss_weight_logret * loss_logret_mix + loss_weight_price_delta * loss_price_delta_mix


def main_train_pytorch_regression(
        start_date_str: str, end_date_str: str,
        initial_features_from_file: Optional[List[str]],
        checkpoint_file_to_use: Path,  # **接收最终的检查点文件名**
        epochs: int, patience: int, batch_size: int,
        lr_base: float, warmup_epochs: int, swa_start_fraction: float,
        d_model_cfg: int, gru_hidden_cfg: int, n_block_cfg: int, gru_layers_cfg: int, n_layer_cfg: int,
        dropout_transformer_cfg: float, dropout_head_cfg: float,
        loss_weight_logret_cfg: float, loss_weight_price_delta_cfg: float
):
    SEED = 42
    random.seed(SEED);
    np.random.seed(SEED);
    torch.manual_seed(SEED)
    if DEVICE.type == 'cuda': torch.cuda.manual_seed_all(SEED)

    logging.info(f"▶ PyTorch 多任务回归训练 (目标: 对数收益率 + Δ价格)")
    # ... (其他日志信息保持不变) ...

    df_processed, a101_scaler_obj = a101.preprocess_data(  # 获取 a101 返回的 scaler 对象
        start_date_str, end_date_str,
        pred_horizon_k_lines=A101_PRED_HORIZON_K,
        agg_period=A101_AGG_PERIOD,
        roll_window_override=A101_ROLL_WINDOW
        # scaler_path_override 会在 a101 内部基于 agg_period 和 roll_window_override 自动生成
    )
    if df_processed.empty:
        logging.error("a101.py 返回了空的 DataFrame。训练终止。")
        return

    # --- 特征选择 ---
    index_name_to_exclude = [df_processed.index.name] if df_processed.index.name else []
    cols_to_exclude = [A101_TARGET_COL, "close_raw", "atr_raw"] + index_name_to_exclude
    all_available_cols = df_processed.columns.tolist()
    actual_features_source_msg = "所有可用特征"

    if initial_features_from_file:
        selected_features = [f for f in initial_features_from_file
                             if f in all_available_cols and f not in cols_to_exclude]
        if not selected_features:
            logging.warning(f"提供的特征文件中的特征均无效或已被排除，将回退使用所有可用特征。")
            selected_features = [c for c in all_available_cols if c not in cols_to_exclude]
            actual_features_source_msg = "所有可用特征 (因文件特征无效)"
        elif len(selected_features) < len(initial_features_from_file):
            filtered_count = len(initial_features_from_file) - len(selected_features)
            logging.warning(f"从特征文件中使用了 {len(selected_features)} 个特征。有 {filtered_count} 个名称被过滤。")
            actual_features_source_msg = f"来自文件 ({len(selected_features)} 个有效)"
        else:
            logging.info(f"成功验证并使用来自特征文件的 {len(selected_features)} 个特征。")
            actual_features_source_msg = f"来自文件 ({len(selected_features)} 个)"
    else:
        selected_features = [c for c in all_available_cols if c not in cols_to_exclude]
        logging.info(f"未使用特征文件，将使用所有可用特征。")
        actual_features_source_msg = "所有可用特征 (未提供文件)"

    if not selected_features:
        logging.error("错误：最终没有选定任何特征用于训练！训练终止。")
        return

    feat_dim = len(selected_features)
    logging.info(f"最终特征来源: {actual_features_source_msg}, 特征维度: {feat_dim}")
    logging.debug(f"最终选定的特征列表: {selected_features}")

    # **关键：基于最终确定的 selected_features 和 feat_dim 来重命名检查点文件**
    # (这个逻辑已移到 if __name__ == "__main__" 中，在调用此函数前确定最终文件名)
    # logging.info(f"最终检查点文件名将是: {checkpoint_file_to_use}")

    required_cols_for_training = [A101_TARGET_COL, 'close_raw', 'atr_raw'] + selected_features
    missing_cols = set(required_cols_for_training) - set(df_processed.columns)
    if missing_cols:
        logging.error(f"错误: DataFrame 中缺少训练/分桶所需列: {missing_cols}。训练终止。")
        return

    actual_processed_data_end_time = df_processed.index.max() if not df_processed.empty else None
    logging.info(f"数据已通过 a101.py 加载。总行数: {len(df_processed)}")
    num_total_rows = len(df_processed)
    min_required_rows = SEQ_LEN + 20  # 至少需要一个完整序列和一些用于测试的额外数据
    if num_total_rows < min_required_rows:
        logging.error(f"处理后的数据行数 ({num_total_rows}) 过少，需要至少 {min_required_rows} 行。训练终止。")
        return

    # --- 数据划分, Dataset, DataLoader ---
    # ... (保持不变)
    split_train_end_idx = int(num_total_rows * 0.7)
    split_val_end_idx = int(num_total_rows * 0.85)
    train_df = df_processed.iloc[:split_train_end_idx].copy()
    val_df = df_processed.iloc[split_train_end_idx:split_val_end_idx].copy()
    test_df = df_processed.iloc[split_val_end_idx:].copy()
    logging.info(f"数据划分: 训练集={len(train_df)}, 验证集={len(val_df)}, 测试集={len(test_df)}")

    cols_for_dataset_df = list(set(selected_features + [A101_TARGET_COL, 'close_raw', 'atr_raw']))
    train_df_pytorch = train_df[cols_for_dataset_df].copy()
    val_df_pytorch = pd.DataFrame()
    if not val_df.empty: val_df_pytorch = val_df[cols_for_dataset_df].copy()
    test_df_pytorch = pd.DataFrame()
    if not test_df.empty: test_df_pytorch = test_df[cols_for_dataset_df].copy()

    train_dataset = PriceRegressionDataset(train_df_pytorch, selected_features, A101_TARGET_COL, SEQ_LEN,
                                           N_VOLATILITY_BUCKETS) if len(train_df_pytorch) >= SEQ_LEN else None
    val_dataset = PriceRegressionDataset(val_df_pytorch, selected_features, A101_TARGET_COL, SEQ_LEN,
                                         n_buckets=1) if val_df_pytorch is not None and len(
        val_df_pytorch) >= SEQ_LEN else None
    test_dataset = PriceRegressionDataset(test_df_pytorch, selected_features, A101_TARGET_COL, SEQ_LEN,
                                          n_buckets=1) if test_df_pytorch is not None and len(
        test_df_pytorch) >= SEQ_LEN else None

    if not train_dataset or len(train_dataset) == 0:
        logging.error("训练数据集创建失败或为空。训练终止。")
        return
    logging.info(
        f"数据集样本数: 训练集={len(train_dataset)}, 验证集={len(val_dataset) if val_dataset else 0}, 测试集={len(test_dataset) if test_dataset else 0}")

    pin_memory_flag = (DEVICE.type == 'cuda')
    train_batch_sampler = BucketBatchSampler(train_dataset.bucket_id, batch_size, shuffle=True, drop_last=False)
    train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=4,
                              pin_memory=pin_memory_flag)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=2,
                            pin_memory=pin_memory_flag) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=2,
                             pin_memory=pin_memory_flag) if test_dataset else None
    logging.info(f"训练 DataLoader 使用桶批次采样器 (实现桶内 MixUp)。验证/测试 DataLoader 使用标准采样。")

    criterion_logret = nn.MSELoss()
    criterion_price_delta = nn.L1Loss()  # 使用 L1Loss 作为 ΔPrice 的基础损失（在两阶段策略中会被 focal_mse 或 smooth_l1 替代）
    logging.info(
        f"损失函数: MSELoss (用于 '{A101_TARGET_COL}'), L1Loss (用于 Δ价格, 两阶段策略中使用 focal_mse/smooth_l1)")

    # --- 模型实例化配置 ---
    current_model_config = {
        'in_dim': feat_dim,  # 使用最终确定的特征维度
        'd_model': d_model_cfg,
        'gru_hidden': gru_hidden_cfg,
        'n_block': n_block_cfg,
        'gru_layers': gru_layers_cfg,
        'n_layer': n_layer_cfg,
        'dropout_transformer': dropout_transformer_cfg,
        'dropout_head': dropout_head_cfg,
        'nhead_transformer': 8  # 保持不变，或也作为参数传入
    }
    model = PricePredictorModel(**current_model_config).to(DEVICE)  # 使用 current_model_config
    logging.info(f"▶ 模型结构 (多任务, 特征维度={feat_dim}):\n{model}\n将在 {DEVICE} 上训练。")
    ema = EMA(model, decay=0.999)  # A201 用更长半衰期

    # ... (优化器、调度器、EMA、SWA设置保持不变) ...
    base_optimizer = optim.AdamW(model.parameters(), lr=lr_base, weight_decay=WEIGHT_DECAY_DEFAULT)
    if epochs > warmup_epochs:
        t_0_main = max(2, int((epochs - warmup_epochs) * 0.12))
        logging.info(f"余弦退火调度器 (CosineAnnealingWarmRestarts) 的 T_0 设置为 {t_0_main} 个 epoch。")
        main_lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            base_optimizer, T_0=t_0_main, T_mult=2, eta_min=lr_base / 100.0)
    else:
        main_lr_scheduler = None

    if warmup_epochs > 0 and main_lr_scheduler is not None:
        warmup_lambda = lambda e: min(1.0, (e + 1) / warmup_epochs)
        warmup_scheduler = optim.lr_scheduler.LambdaLR(base_optimizer, lr_lambda=warmup_lambda)
        lr_scheduler = optim.lr_scheduler.SequentialLR(base_optimizer, schedulers=[warmup_scheduler, main_lr_scheduler],
                                                       milestones=[warmup_epochs])
    elif warmup_epochs > 0:
        warmup_lambda = lambda e: min(1.0, (e + 1) / warmup_epochs)
        lr_scheduler = optim.lr_scheduler.LambdaLR(base_optimizer, lr_lambda=warmup_lambda)
    elif main_lr_scheduler is not None:
        lr_scheduler = main_lr_scheduler
    else:
        lr_scheduler = optim.lr_scheduler.ConstantLR(base_optimizer, factor=1.0)

    grad_scaler = GradScaler(enabled=(DEVICE.type == 'cuda'))
    ema_model = EMA(model, decay=0.999)

    swa_model = optim.swa_utils.AveragedModel(model)
    swa_lr_scheduler = optim.swa_utils.SWALR(base_optimizer, swa_lr=lr_base / 20.0)
    swa_start_epoch = max(warmup_epochs + 1, int(epochs * swa_start_fraction)) if epochs > warmup_epochs else epochs + 1

    # --- 训练循环 ---
    # ... (训练循环内部逻辑保持不变，除了 checkpoint 保存) ...
    best_val_composite_metric = float('inf')
    epochs_no_improve = 0
    logging.info(f"开始训练，共 {epochs} 个 epoch...")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_train_loss, num_train_samples_processed = 0.0, 0
        batch_log_counter = 0
        if not train_loader: break
        for x_batch, y_batch_log_return, current_close_raw_batch in train_loader:
            x_batch = x_batch.to(DEVICE);
            y_batch_log_return = y_batch_log_return.to(DEVICE);
            current_close_raw_batch = current_close_raw_batch.to(DEVICE)
            mixed_x, (y_a_logret, y_b_logret, y_a_delta, y_b_delta, _, _, lam) = \
                mixup_data_regression(x_batch, y_batch_log_return, current_close_raw_batch,
                                      alpha=0.0)  # alpha=0 means no mixup in training loop for simplicity now
            base_optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=DEVICE.type, enabled=(DEVICE.type == 'cuda')):
                pred_log_return, pred_price_delta_raw = model(mixed_x)
                pred_price_delta = dprice_encode(pred_price_delta_raw)
                if epoch < 5:
                    loss_dp = focal_mse(pred_price_delta, y_a_delta * lam + y_b_delta * (1 - lam))
                    loss_lr = criterion_logret(pred_log_return, y_a_logret * lam + y_b_logret * (1 - lam)) * 0.5
                else:
                    loss_dp = smooth_l1(pred_price_delta, y_a_delta * lam + y_b_delta * (
                                1 - lam)) * loss_weight_price_delta_cfg  # Use configured weight
                    loss_lr = criterion_logret(pred_log_return, y_a_logret * lam + y_b_logret * (
                                1 - lam)) * loss_weight_logret_cfg  # Use configured weight
                loss = loss_dp + loss_lr
            if torch.isnan(loss) or torch.isinf(loss):
                logging.error(f"Epoch {epoch}, 批次: 检测到 NaN/Inf 损失。跳过此批次。");
                continue
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(base_optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_scaler.step(base_optimizer);
            grad_scaler.update()
            ema_model.update(model)
            epoch_train_loss += loss.item() * mixed_x.size(0)
            num_train_samples_processed += mixed_x.size(0)
            batch_log_counter += 1
            if batch_log_counter % 50 == 0: logging.debug(
                f"Epoch {epoch}, 批次 {batch_log_counter}/{len(train_loader)}, 瞬时损失: {loss.item():.6f}")

        avg_epoch_train_loss = epoch_train_loss / num_train_samples_processed if num_train_samples_processed > 0 else float(
            'nan')
        current_lr_for_log = base_optimizer.param_groups[0]['lr']
        if epoch > swa_start_epoch:
            swa_model.update_parameters(model)
            if epoch == swa_start_epoch + 1: logging.info(f"SWA 在 Epoch {epoch} 开始。切换到 SWALR 学习率调度器。")
            swa_lr_scheduler.step()
        else:
            lr_scheduler.step()

        val_combined_loss_ema, val_composite_metric_ema = float('nan'), float('inf')
        val_mae_logret_ema, val_rmse_logret_ema = float('nan'), float('nan')
        val_mae_delta_price_ema, val_rmse_delta_price_ema = float('nan'), float('nan')
        val_mae_price_scale_ema, val_rmse_price_scale_ema = float('nan'), float('nan')

        if val_loader and val_dataset and len(val_dataset) > 0:
            ema_model.apply_shadow(model)  # 应用EMA权重进行验证
            (val_combined_loss_ema, val_composite_metric_ema,
             val_mae_logret_ema, val_rmse_logret_ema,
             val_mae_delta_price_ema, val_rmse_delta_price_ema,
             val_mae_price_scale_ema, val_rmse_price_scale_ema,
             _, _, _, _, _) = evaluate_log_return_regression(
                model, val_loader, criterion_logret, criterion_price_delta,  # 使用 criterion_price_delta (L1Loss)
                loss_weight_logret_cfg, loss_weight_price_delta_cfg
            )
            model.train()  # 验证后切回训练模式
            log_msg = (f"Epoch {epoch:03d}/{epochs} | LR: {current_lr_for_log:.2e} | "
                       f"训练损失(组合): {avg_epoch_train_loss:.6f} | "
                       f"验证EMA损失(组合): {val_combined_loss_ema:.6f} | "
                       f"验证EMA复合指标: {val_composite_metric_ema:.4f} | "
                       f"RMSE(对数收益率): {val_rmse_logret_ema:.6f} | RMSE(Δ价格): {val_rmse_delta_price_ema:.2f} | RMSE(价格尺度): {val_rmse_price_scale_ema:.2f}")
            logging.info(log_msg)
        else:
            logging.info(
                f"Epoch {epoch:03d}/{epochs} | LR: {current_lr_for_log:.2e} | 训练损失(组合): {avg_epoch_train_loss:.6f} | 无验证集。")
            val_composite_metric_ema = avg_epoch_train_loss  # 如果没有验证集，用训练损失代替（不推荐，但避免崩溃）

        if np.isnan(avg_epoch_train_loss) or np.isnan(val_composite_metric_ema):
            logging.error(f"Epoch {epoch}: 检测到 NaN 损失/指标。提前停止训练。");
            break

        if val_composite_metric_ema < best_val_composite_metric:
            best_val_composite_metric = val_composite_metric_ema
            epochs_no_improve = 0
            swa_state_to_save = None
            if epoch > swa_start_epoch:
                try:
                    logging.info(f"保存检查点前，使用训练数据更新 SWA 模型 BN 统计量...")
                    torch.optim.swa_utils.update_bn(train_loader, swa_model, device=DEVICE)
                    swa_state_to_save = swa_model.module.state_dict()  # 获取SWA模型权重
                    logging.info("SWA BN 统计量已更新。")
                except Exception as e_bn:
                    logging.error(f"更新 SWA BN 统计量时出错: {e_bn}。SWA 模型状态可能不是最优。")
                    if hasattr(swa_model, 'module'): swa_state_to_save = swa_model.module.state_dict()  # 仍然尝试保存

            # **确保所有需要的元数据都被保存**
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),  # 基础模型权重
                "ema_shadow_params": ema_model.shadow_params,  # EMA权重
                "swa_model_state_dict": swa_state_to_save,  # SWA模型权重 (可能为None)
                "optimizer_state_dict": base_optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "swa_lr_scheduler_state_dict": swa_lr_scheduler.state_dict() if epoch > swa_start_epoch else None,
                "grad_scaler_state_dict": grad_scaler.state_dict(),  # 保存GradScaler状态
                "best_val_metric_name": "val_composite_metric_ema (0.7*RMSE_Px + 0.3*RMSE_LogRet)",
                "best_val_metric_value": best_val_composite_metric,
                "val_metrics_at_save": {  # 保存当时的验证集指标
                    "combined_loss": val_combined_loss_ema, "composite_metric": val_composite_metric_ema,
                    "mae_logret": val_mae_logret_ema, "rmse_logret": val_rmse_logret_ema,
                    "mae_delta_price": val_mae_delta_price_ema, "rmse_delta_price": val_rmse_delta_price_ema,
                    "mae_price_scale": val_mae_price_scale_ema, "rmse_price_scale": val_rmse_price_scale_ema,
                },
                "feat_dim": feat_dim,  # 实际使用的特征数量
                "selected_features_names": selected_features,  # 实际使用的特征名列表
                "input_cols": selected_features,
                "input_dim": feat_dim,

                "model_config": current_model_config,  # 实例化模型时用的配置
                "a101_config": {  # a101.py 处理数据时的配置
                    "pred_horizon_k": A101_PRED_HORIZON_K,
                    "agg_period": A101_AGG_PERIOD,
                    "roll_window": A101_ROLL_WINDOW,
                    "target_col": A101_TARGET_COL
                },
                "loss_weights": {  # 损失权重
                    "logret": loss_weight_logret_cfg,
                    "delta_price": loss_weight_price_delta_cfg
                },
                # a101.py 使用的 scaler 文件路径 (它在 a101.preprocess_data 中基于参数动态生成或加载)
                # 这里我们假设 a101.SCALER_PATH 是其内部使用的最终路径对象 (如果它是一个全局变量或属性)
                # 如果 a101_scaler_obj 是返回的 scaler 对象，则其 .scaler_path (如果该属性存在)
                "a101_scaler_path": str(getattr(a101_scaler_obj, 'scaler_path_used_for_fit', Path(
                    a101.SCALER_PATH.parent) / f"scaler_{A101_AGG_PERIOD}_rw{A101_ROLL_WINDOW}.joblib").resolve()),
                "processed_data_actual_end_time": str(
                    actual_processed_data_end_time) if actual_processed_data_end_time else None,
                "training_args": {  # 保存命令行参数，便于复现
                    "start_date": start_date_str, "end_date": end_date_str, "epochs": epochs, "patience": patience,
                    "batch_size": batch_size, "lr_base": lr_base, "warmup_epochs": warmup_epochs,
                    "swa_start_fraction": swa_start_fraction, "d_model_cfg": d_model_cfg,
                    "gru_hidden_cfg": gru_hidden_cfg, "n_block_cfg": n_block_cfg, "gru_layers_cfg": gru_layers_cfg,
                    "n_layer_cfg": n_layer_cfg, "dropout_transformer_cfg": dropout_transformer_cfg,
                    "dropout_head_cfg": dropout_head_cfg, "loss_weight_logret_cfg": loss_weight_logret_cfg,
                    "loss_weight_price_delta_cfg": loss_weight_price_delta_cfg
                }
            }
            torch.save(checkpoint_data, checkpoint_file_to_use)
            logging.info(
                f"✔ 检查点已保存: Epoch {epoch}, 验证EMA复合指标: {best_val_composite_metric:.4f} -> {checkpoint_file_to_use}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logging.warning(
                    f"✖ 早停触发于 Epoch {epoch} (复合指标连续 {patience} 个 epoch 未改善)。最佳指标: {best_val_composite_metric:.4f}");
                break

    # --- 训练结束，最终评估 ---
    if checkpoint_file_to_use.exists():
        logging.info(f"▶ 加载最佳检查点进行最终评估: {checkpoint_file_to_use}")
        checkpoint = torch.load(checkpoint_file_to_use, map_location=DEVICE,
                                weights_only=False)  # weights_only=False 以加载所有元数据

        # 优先使用检查点中的模型配置和特征信息
        loaded_model_config = checkpoint.get("model_config")
        if loaded_model_config is None:
            logging.warning("检查点缺少 'model_config' 信息，将使用当前脚本的默认模型参数！这可能导致错误！")
            loaded_model_config = current_model_config  # Fallback to current config

        loaded_feat_dim = checkpoint.get("feat_dim")
        loaded_features_names = checkpoint.get("selected_features_names")

        if loaded_feat_dim is None or loaded_features_names is None:
            logging.error(f"检查点缺少特征信息 (feat_dim 或 selected_features_names)！无法继续评估。")
            return
        logging.info(f"检查点模型使用 {loaded_feat_dim} 个特征进行训练: {loaded_features_names[:5]}...")

        # --- 评估 EMA 模型 ---
        logging.info("评估来自检查点的最佳 EMA 模型...")
        model_ema_eval = PricePredictorModel(**loaded_model_config).to(DEVICE)  # 使用检查点中的配置
        final_ema = EMA(model_ema_eval, decay=EMA_DECAY)  # 用当前脚本的EMA_DECAY
        if "ema_shadow_params" in checkpoint and checkpoint["ema_shadow_params"] is not None:
            final_ema.shadow_params = checkpoint["ema_shadow_params"]
            final_ema.apply_shadow(model_ema_eval)
            logging.info("已应用检查点中的 EMA 权重到评估模型。")
        elif "model_state_dict" in checkpoint:
            logging.warning("检查点中未找到 EMA 影子参数。将评估检查点中的基础模型状态。")
            model_ema_eval.load_state_dict(checkpoint["model_state_dict"])
        else:
            logging.error("检查点既无EMA参数也无基础模型状态，无法评估。")
            return

        # ... (后续的验证指标打印和测试集评估逻辑保持不变) ...
        saved_metrics = checkpoint.get("val_metrics_at_save", {})
        log_best_val_metric = checkpoint.get('best_val_metric_value', 'N/A')
        if isinstance(log_best_val_metric, float): log_best_val_metric = f"{log_best_val_metric:.4f}"
        logging.info(f"检查点来自 Epoch {checkpoint['epoch']}。保存的最佳验证复合指标: {log_best_val_metric}")
        logging.info(f"  保存时的验证指标: {saved_metrics}")

        if test_loader and test_dataset and len(test_dataset) > 0:
            logging.info("在测试集上评估最佳 EMA 模型...")
            (test_loss_comb_ema, test_comp_metric_ema,
             test_mae_logret_ema, test_rmse_logret_ema,
             test_mae_dprice_ema, test_rmse_dprice_ema,
             test_mae_px_ema, test_rmse_px_ema,
             _, _, _, _, _) = evaluate_log_return_regression(
                model_ema_eval, test_loader, criterion_logret, criterion_price_delta,
                loss_weight_logret_cfg, loss_weight_price_delta_cfg
            )
            logging.info(f"◎ EMA 测试集结果:")
            logging.info(f"  组合损失       : {test_loss_comb_ema:.6f}")
            logging.info(f"  复合指标       : {test_comp_metric_ema:.4f}")
            logging.info(f"  对数收益率指标 : MAE={test_mae_logret_ema:.6f}, RMSE={test_rmse_logret_ema:.6f}")
            logging.info(f"  Δ价格 指标     : MAE={test_mae_dprice_ema:.2f}, RMSE={test_rmse_dprice_ema:.2f}")
            logging.info(f"  价格尺度 指标  : MAE={test_mae_px_ema:.2f}, RMSE={test_rmse_px_ema:.2f}")
        else:
            logging.warning("测试集为空或不可用。跳过最终 EMA 测试评估。")

        # --- 评估 SWA 模型 (如果存在) ---
        if checkpoint.get("swa_model_state_dict") is not None:
            logging.info("评估来自检查点的最佳 SWA 模型...")
            model_swa_eval = PricePredictorModel(**loaded_model_config).to(DEVICE)
            model_swa_eval.load_state_dict(checkpoint["swa_model_state_dict"])
            logging.info("使用训练数据更新 SWA 模型 BatchNorm 统计量...")
            if train_loader:  # 确保 train_loader 仍然可用
                try:
                    torch.optim.swa_utils.update_bn(train_loader, model_swa_eval, device=DEVICE)
                    logging.info("SWA BN 统计量已更新。")
                except Exception as e_bn_final:
                    logging.error(f"最终评估时更新 SWA BN 统计量出错: {e_bn_final}")
            else:
                logging.warning("训练 DataLoader 不可用，无法为最终评估更新 SWA BN 统计量。")

            if test_loader and test_dataset and len(test_dataset) > 0:
                logging.info("在测试集上评估 SWA 模型...")
                (test_loss_comb_swa, test_comp_metric_swa, _, _, _, _, _, _,
                 _, _, _, _, _) = evaluate_log_return_regression(
                    model_swa_eval, test_loader, criterion_logret, criterion_price_delta,
                    loss_weight_logret_cfg, loss_weight_price_delta_cfg
                )
                logging.info(f"◎ SWA 测试集结果:")
                logging.info(f"  组合损失       : {test_loss_comb_swa:.6f}")
                logging.info(f"  复合指标       : {test_comp_metric_swa:.4f}")
        else:
            logging.info("检查点中未找到 SWA 模型状态，或未启用 SWA。")
    else:
        logging.warning(f"检查点文件 {checkpoint_file_to_use} 未找到。跳过最终评估。")
    logging.info("多任务回归训练与评估流程结束。")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch 多任务价格回归模型 (对数收益率+Δ价格) 训练脚本 (带桶内MixUp)")
    parser.add_argument("--start", default=START_DATE_DEFAULT, help="数据开始日期时间 (格式: YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--end", default=END_DATE_DEFAULT, help="数据请求结束日期时间")
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT, help="训练总轮数")
    parser.add_argument("--patience", type=int, default=PATIENCE_DEFAULT, help="早停判断的耐心轮数 (基于复合指标)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE_DEFAULT, help="批次大小 (用于桶批次采样器)")
    parser.add_argument("--lr", type=float, default=LR_BASE_DEFAULT, help="基础学习率")
    parser.add_argument("--warmup", type=int, default=WARMUP_EPOCHS_DEFAULT, help="Warmup 轮数")
    parser.add_argument("--swa_frac", type=float, default=SWA_START_FRAC_DEFAULT,
                        help="SWA 开始的 epoch 比例 (相对于总轮数)")
    parser.add_argument("--wd", type=float, default=WEIGHT_DECAY_DEFAULT, help="AdamW 权重衰减")
    parser.add_argument("--d_model_cfg", type=int, default=D_MODEL_DEFAULT, help="模型内部维度 d_model")
    parser.add_argument("--gru_hidden_cfg", type=int, default=GRU_HIDDEN_DEFAULT, help="GRU 隐藏层大小 (单向)")
    parser.add_argument("--n_block_cfg", type=int, default=N_BLOCK_DEFAULT, help="TimesBlock 块数量")
    parser.add_argument("--gru_layers_cfg", type=int, default=GRU_LAYERS_DEFAULT, help="GRU 层数")
    parser.add_argument("--n_layer_cfg", type=int, default=N_LAYER_DEFAULT, help="Transformer Encoder 层数")
    parser.add_argument("--dropout_transformer_cfg", type=float, default=DROPOUT_TRANSFORMER_LAYER_DEFAULT,
                        help="Transformer 层 Dropout")
    parser.add_argument("--dropout_head_cfg", type=float, default=DROPOUT_MODEL_HEAD_DEFAULT, help="模型头部 Dropout")
    parser.add_argument("--lgbm_features_file", type=str, default=LGBM_FEATURE_FILE_DEFAULT,
                        help=f"指定包含特征列表的文本文件名。若此文件存在且有效，则使用；否则使用所有特征。默认: {LGBM_FEATURE_FILE_DEFAULT}")
    parser.add_argument("--delete_checkpoint", action="store_true", help="开始训练前删除旧检查点文件。")
    parser.add_argument("--loss_weight_logret", type=float, default=LOSS_WEIGHT_LOGRET,
                        help="对数收益率损失(MSE)的权重")
    parser.add_argument("--loss_weight_dprice", type=float, default=LOSS_WEIGHT_DELTA_PRICE,
                        help="价格变化损失(MAE)的权重")
    args = parser.parse_args()

    feature_filepath = Path(args.lgbm_features_file)
    initial_features_from_file = None
    num_features_from_file_str = "all"  # 用于文件名后缀

    if feature_filepath.exists() and feature_filepath.is_file():
        try:
            with open(feature_filepath, "r") as f:
                features_in_file = [line.strip() for line in f if line.strip()]
            if features_in_file:
                initial_features_from_file = features_in_file
                num_features_from_file_str = f"lgbm{len(features_in_file)}"
                logging.info(f"找到特征文件: {feature_filepath}。包含 {len(features_in_file)} 个特征名，将尝试使用。")
            else:
                logging.warning(f"特征文件 {feature_filepath} 存在但为空。将使用所有可用特征。")
        except Exception as e:
            logging.error(f"读取特征文件 {feature_filepath} 时出错: {e}。将使用所有可用特征。")
    else:
        logging.info(f"特征文件 {feature_filepath} 未找到。将使用所有可用特征。")

    # **在调用 main_train_pytorch_regression 之前确定最终的检查点文件名**
    # 这个文件名现在将准确反映实际使用的特征情况 (来自文件或 'all')
    checkpoint_suffix_feat_final = num_features_from_file_str  # 使用上面确定的后缀
    checkpoint_suffix_task = f"mt_logret_dprice_K{A101_PRED_HORIZON_K}_agg{A101_AGG_PERIOD}_rw{A101_ROLL_WINDOW}_bktmix"
    final_checkpoint_file = Path(f"best_ckpt_pytorch_{checkpoint_suffix_feat_final}_{checkpoint_suffix_task}.pth")

    if args.delete_checkpoint and final_checkpoint_file.exists():
        logging.warning(f"根据命令行参数，删除已存在的检查点文件: {final_checkpoint_file}")
        try:
            final_checkpoint_file.unlink()
        except OSError as e:
            logging.error(f"删除检查点文件时出错: {e}")

    logging.info("--- 最终训练配置 ---")
    # ... (其他日志打印)
    logging.info(f"  特征来源指示   : {num_features_from_file_str.upper()}")  # 显示 lgbm<N> 或 ALL
    if initial_features_from_file: logging.info(f"    文件路径     : {feature_filepath}")
    logging.info(f"  最终检查点文件 : {final_checkpoint_file}")
    # ...

    main_train_pytorch_regression(
        start_date_str=args.start,
        end_date_str=args.end,
        initial_features_from_file=initial_features_from_file,
        checkpoint_file_to_use=final_checkpoint_file,  # **传递最终确定的文件名**
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr_base=args.lr,
        warmup_epochs=args.warmup,
        swa_start_fraction=args.swa_frac,
        d_model_cfg=args.d_model_cfg,
        gru_hidden_cfg=args.gru_hidden_cfg,
        n_block_cfg=args.n_block_cfg,
        gru_layers_cfg=args.gru_layers_cfg,
        n_layer_cfg=args.n_layer_cfg,
        dropout_transformer_cfg=args.dropout_transformer_cfg,
        dropout_head_cfg=args.dropout_head_cfg,
        loss_weight_logret_cfg=args.loss_weight_logret,
        loss_weight_price_delta_cfg=args.loss_weight_dprice
    )