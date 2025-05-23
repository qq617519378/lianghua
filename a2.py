#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MODULE 2: 混合神经微分方程 (Hybrid Neural ODE) + 小波分解 - 调用模块1数据处理 (使用 dopri5)
---------------------------------------------------------------------------
功能:
  1. 从 a1.py (NewDataProcessor) 获取数据库加密货币数据, 进行预处理与归一化
  2. 构建 TimeseriesWaveletDataset（可含事件列），按滑动窗口切分
  3. 小波分解(GPU 优先) + Neural ODE (torchdiffeq - dopri5) 做价格回归预测
  4. 训练过程中加入梯度裁剪、Early Stopping、StepLR 调度，并输出训练/验证/测试的 MSE/RMSE
  5. 将最佳模型权重和优化器状态保存（checkpoint_best.pth / best_model.pth），
     并在 checkpoint 中额外记录本次训练数据的截止时间 (train_end_timestamp)，
     供后续在线学习或增量训练使用。

用法:
  python module2_hybrid_ode.py

依赖:
  pip install torch torchdiffeq pytorch-wavelets numpy pandas scikit-learn tqdm
"""
from core_utils import EMA
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import logging
logging.getLogger("torch").setLevel(logging.ERROR)
import numpy as np
import pandas as pd


import torch
import torch.nn as nn
import torch.optim as optim
import torch
torch.backends.cudnn.benchmark = True


from sklearn.preprocessing import StandardScaler
import  os
from torch.cuda.amp import GradScaler, autocast

try:
    from pytorch_wavelets import DWT1DForward
    HAS_PYTORCH_WAVELETS = True
except ImportError:
    HAS_PYTORCH_WAVELETS = False

import pywt  # CPU小波备用

try:
    from torchdiffeq import odeint
except ImportError:
    raise ImportError("请 pip install torchdiffeq 以使用 Neural ODE 求解器")
# 在文件顶部
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
BATCH_SIZE = 1024   # 先试512，如果显存还够，再考虑更大

use_amp = False
# 在文件最顶部（或适当位置）定义正常梯度范围
from collections import OrderedDict  # EMA 用

# ---- 模块1: a1.py 引用 NewDataProcessor ----
from a1 import NewDataProcessor
from config import LOSS_WEIGHT_DELTA_PRICE
import config

import math
import torch.nn.functional as F
# -------------------------------------------------
# 自实现 numerically‑stable log‑cosh
def log_cosh(x: torch.Tensor):
    return x + F.softplus(-2.0 * x) - math.log(2.0)
# -------------------------------------------------


###############################################################################
# 日志设置
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)

###############################################################################
# 1) WaveletTransform（GPU/CPU）
###############################################################################
# ---------- EMA 工具 ----------
def weighted_loss(pred, target):
    # Huber beta=0.5
    huber_part = nn.SmoothL1Loss(beta=0.5, reduction='none')(pred, target)
    # 自实现 log-cosh
    logcosh_part = log_cosh(pred - target)
    # 给 |y| > std 的大波动样本加倍权重
    weight = 1.0 + (target.abs() > target.std()).float()
    return (huber_part + 0.2 * logcosh_part) * weight






class GPUWaveletTransform(nn.Module):
    """GPU版小波分解(基于 pytorch_wavelets)"""
    def __init__(self, wave='db4', level=3, mode='zero'):
        super().__init__()
        if not HAS_PYTORCH_WAVELETS:
            raise RuntimeError("pytorch-wavelets 未安装, 无法使用 GPUWaveletTransform")
        self.dwt = DWT1DForward(wave=wave, J=level, mode=mode)

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, F)
        returns:
            low_freq:  (B, L_low, F)
            multi_hf:  [ (B, L_hf, F), ... ]
        """
        x_perm = x.permute(0,2,1).contiguous()  # (B, F, T)
        approx, details = self.dwt(x_perm)
        low_freq = approx.permute(0,2,1).contiguous()
        multi_hf = [d.permute(0,2,1).contiguous() for d in details]
        return low_freq, multi_hf


class CPUWaveletTransform(nn.Module):
    """CPU版小波分解(基于 pywt)"""
    def __init__(self, wave='db4', level=3):
        super().__init__()
        self.wave = wave
        self.level = level

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, F)
        returns:
            low_freq:  (B, L_low, F)
            multi_hf:  [ (B, L_hf, F), ... ]
        """
        B, T, F = x.shape
        x_np = x.cpu().numpy()
        low_list = []
        hf_lists = [[] for _ in range(self.level)]
        for b in range(B):
            low_batch = []
            hf_batch = [[] for _ in range(self.level)]
            for f_ in range(F):
                coeffs = pywt.wavedec(x_np[b, :, f_], self.wave, level=self.level)
                low_batch.append(coeffs[0])
                for i, cd in enumerate(coeffs[1:]):
                    hf_batch[i].append(cd)
            low_batch = np.stack(low_batch, axis=1)  # shape (L_low, F)
            for i in range(self.level):
                hf_batch[i] = np.stack(hf_batch[i], axis=1)
            low_list.append(low_batch)
            for i in range(self.level):
                hf_lists[i].append(hf_batch[i])
        low_freq = torch.tensor(np.array(low_list), dtype=torch.float32)
        multi_hf = [torch.tensor(np.array(hf_lists[i]), dtype=torch.float32) for i in range(self.level)]
        return low_freq, multi_hf


###############################################################################
# 2) Neural ODE
###############################################################################
class ODEFunc(nn.Module):
    """
    ODEFunc: f(t, h) -> dh/dt
    - 两层 MLP (SiLU + spectral_norm可选) + clamp
    """
    def __init__(self, hidden_dim=64, clamp_val=10.0, spectral=True):
        super().__init__()
        def maybe_sn(linear):
            return nn.utils.spectral_norm(linear) if spectral else linear

        layers = []
        layers.append(maybe_sn(nn.Linear(hidden_dim, hidden_dim)))
        layers.append(nn.SiLU())
        layers.append(maybe_sn(nn.Linear(hidden_dim, hidden_dim)))
        self.net = nn.Sequential(*layers)
        self.clamp_val = clamp_val

    def forward(self, t, h):
        dh = self.net(h)
        if self.clamp_val:
            dh = torch.clamp(dh, -self.clamp_val, self.clamp_val)
        return dh


class ODEBlock(nn.Module):
    """
    ODEBlock: 这里使用 torchdiffeq 的 'dopri5' 方法在 GPU 上求解
    """

    def __init__(self, odefunc, method='dopri5', rtol=1e-5, atol=1e-6,
                               max_step=None, step_size=None):  # dopri5 自适应
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        # 强制使用 RK4 固定步长，避免 dopri5 自适应步长 underflow
        self.method = 'rk4'
        self.rtol = rtol  # 对 RK4 无影响，可留作记录
        self.atol = atol
        self.max_step = max_step
        # 如果外部没有传步长，就用 0.05
        self.step_size = step_size if step_size is not None else 0.05

        self.register_buffer('integration_time',
                             torch.tensor([0.0, 1.0]))

    def forward(self, x):
        t = self.integration_time.to(x.device)
        options = {}
        if self.step_size is not None:
            options = {'step_size': self.step_size}
        if self.max_step is not None:
            options['max_num_steps'] = self.max_step

        out = odeint(
            self.odefunc,
            x,
            t,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            options=options if options else None
        )
        return out[-1]


###############################################################################
# 3) EventEncoder（可选）
###############################################################################
class EventEncoder(nn.Module):
    """
    (B, T, E) -> (B, T, enc_dim) -> time_mean => (B, enc_dim)
    """
    def __init__(self, event_dim, enc_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(event_dim, 64),
            nn.ReLU(),
            nn.Linear(64, enc_dim)
        )

    def forward(self, evt):
        B, T, E = evt.shape
        x = evt.view(B*T, E)
        out = self.net(x)
        out = out.view(B, T, -1)
        return out


###############################################################################
# 4) HybridNeuralODEModel: Wavelet + ODE + 事件(可选)
###############################################################################
class HybridNeuralODEModel(nn.Module):
    """
    1. input_proj -> wavelet -> mean(低频/多级高频 + time_mean)
    2. event encoder(可选) -> mean
    3. fusion -> ODEBlock -> readout => 预测 (回归)
    """

    def __init__(self,
                 input_dim,
                 proj_dim=64,
                 hidden_dim=64,
                 wave='db4',
                 level=3,
                 use_gpu_wavelet=True,
                 event_dim=0,
                 clamp_val=10.0,
                 spectral=True,
                 rtol=1e-2,
                 atol=1e-2,
                 max_step=None,
                 ode_method = 'rk4',
                 ode_step = 0.1):
        super().__init__()
        self.proj_dim = proj_dim
        self.hidden_dim = hidden_dim
        self.event_dim = event_dim

        # 输入投影
        self.input_proj = nn.Linear(input_dim, proj_dim)

        # wavelet
        if use_gpu_wavelet:
            if not HAS_PYTORCH_WAVELETS:
                raise RuntimeError("pytorch-wavelets 未安装, 无法使用 GPUWaveletTransform")
            self.wavelet = GPUWaveletTransform(wave=wave, level=level, mode='zero')
        else:
            self.wavelet = CPUWaveletTransform(wave=wave, level=level)

        # 事件编码
        if event_dim > 0:
            self.event_encoder = EventEncoder(event_dim, enc_dim=hidden_dim//2)
        else:
            self.event_encoder = None

        # wave分解后：低频+level个高频 => (level+1)*proj_dim
        # + 原序列的 time_mean => +proj_dim => (level+2)*proj_dim
        fusion_in_dim = (level + 2)*proj_dim
        if self.event_encoder is not None:
            fusion_in_dim += (hidden_dim // 2)

        self.fusion = nn.Linear(fusion_in_dim, hidden_dim)

        # Neural ODE
        self.odefunc = ODEFunc(hidden_dim=hidden_dim,
                               clamp_val=clamp_val,
                               spectral=spectral)
        self.odeblock = ODEBlock(
            self.odefunc,
            # method 参数已被内部重写为 'rk4'
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            step_size=ode_step  # 推荐 0.05，可调 0.02~0.1
        )

        # 回归输出
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )

    def forward(self, x_seq, evt_seq=None):
        """
        x_seq: (B, T, input_dim)
        evt_seq: (B, T, event_dim) or None
        """
        # 投影
        proj = self.input_proj(x_seq)  # (B, T, proj_dim)

        # wavelet 分解
        #with amp.autocast('cuda', enabled=False):  # 指明 device_type，且关闭混合精度

        with autocast( enabled=False):
            low_freq, multi_hf = self.wavelet(proj.float())

        low_mean = low_freq.mean(dim=1)            # (B, proj_dim)
        hf_means = [hf.mean(dim=1) for hf in multi_hf]  # (B, proj_dim) * level
        time_mean = proj.mean(dim=1)               # (B, proj_dim)

        # 事件序列编码
        if self.event_encoder and evt_seq is not None and evt_seq.size(-1) > 0:
            evt_out = self.event_encoder(evt_seq)
            evt_mean = evt_out.mean(dim=1)  # (B, enc_dim)
        else:
            evt_mean = None

        # 融合特征
        feats = [low_mean] + hf_means + [time_mean]
        fused_in = torch.cat(feats, dim=-1)
        if evt_mean is not None:
            fused_in = torch.cat([fused_in, evt_mean], dim=-1)

        # ODE过程
        h0 = self.fusion(fused_in)  # h0 仍是半精度 (FP16) in autocast
        # ── 让 ODEBlock 在 FP32 下安全积分 ─────────────────────────
        with autocast(enabled=False):
            h_final = self.odeblock(h0.float())  # 转成 FP32 进入 ODE
        h_final = h_final.to(h0.dtype)  # 再转回 FP16，保持流水线一致

        # 回归输出
        out = self.readout(h_final)    # (B, 1)
        return out.squeeze(-1)         # (B,)


###############################################################################
# 5) TimeseriesWaveletDataset: 回归任务
###############################################################################
class TimeseriesWaveletDataset(Dataset):
    """
    从 df 中构建滑动窗口来预测未来 (horizon) 的目标值 (target_col)。
    """
    def __init__(self, df, input_cols, target_col,
                 seq_len=120, horizon=1, event_cols=None):
        super().__init__()
        self.input_cols = input_cols
        self.target_col = target_col
        self.event_cols = event_cols if event_cols else []
        self.seq_len = seq_len
        self.horizon = horizon

        # 按时间排序
        df_sorted = df.sort_values('time_window').reset_index(drop=True)
        self.X_data = df_sorted[input_cols].values
        self.y_data = df_sorted[target_col].values
        self.n_data = len(df_sorted)

        if self.event_cols:
            valid_evt = [c for c in self.event_cols if c in df_sorted.columns]
            self.ev_data = df_sorted[valid_evt].values if valid_evt else None
        else:
            self.ev_data = None

        self.max_start = self.n_data - seq_len - self.horizon + 1

    def __len__(self):
        return max(0, self.max_start)

    def __getitem__(self, idx):
        i_start = idx
        i_end = idx + self.seq_len
        i_tar = i_end + self.horizon - 1

        x_seq = self.X_data[i_start:i_end]  # (seq_len, input_dim)
        y_val = self.y_data[i_tar]

        if self.ev_data is not None:
            ev_seq_np = self.ev_data[i_start:i_end]
            ev_seq = torch.tensor(ev_seq_np, dtype=torch.float32)
        else:
            ev_seq = torch.empty((self.seq_len, 0), dtype=torch.float32)

        return {
            'x_seq': torch.tensor(x_seq, dtype=torch.float32),
            'y_val': torch.tensor([y_val], dtype=torch.float32),
            'event_seq': ev_seq
        }


###############################################################################
# 6) 训练 & 验证
###############################################################################
def train_one_epoch(model, loader, optimizer, criterion, clip_grad, device, ema):
    model.train()
    total_loss, total_count = 0.0, 0
    pbar = tqdm(loader, desc="Train", dynamic_ncols=True, leave=False)
    for batch in pbar:
        # —— 准备输入 ——
        tmp_x = batch['x_seq']
        # 可选的 NaN 检测（之前已加过，如果不需要可删）
        if torch.isnan(tmp_x).any() or torch.isinf(tmp_x).any():
            raise ValueError("train_one_epoch: 输入含 NaN/Inf，停止训练")
        x_seq = tmp_x.to(device, non_blocking=True)
        y_val = batch['y_val'].to(device, non_blocking=True)
        evt_seq = batch['event_seq'].to(device, non_blocking=True) if batch['event_seq'].size(-1) > 0 else None

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=False):
            pred = model(x_seq, evt_seq)
            loss = criterion(pred, y_val.view(-1))

        # —— 反向 & 梯度裁剪 ——
        # —— 反向（普通 FP32） & 梯度裁剪 ——
        loss.backward()
        if clip_grad:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

        # —— 5) 更新参数 ——
        optimizer.step()
        ema.update(model)

        # 统计 loss
        bs = y_val.size(0)
        total_loss += loss.item() * bs
        total_count += bs

    return total_loss / total_count



def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            x_seq = batch['x_seq'].to(device, non_blocking=True)
            y_val = batch['y_val'].to(device, non_blocking=True)
            evt_seq = batch['event_seq'].to(device, non_blocking=True) if batch['event_seq'].size(-1) > 0 else None

            pred = model(x_seq, evt_seq)
            loss = criterion(pred, y_val.view(-1))
            bs = y_val.size(0)
            total_loss += loss.item() * bs
            total_count += bs

    return total_loss / total_count


###############################################################################
# 7) 主入口 (带记录训练数据截止时间)
###############################################################################
def main():
    logging.info("=== 模块2：Hybrid Neural ODE (价格预测) - 使用 dopri5 ===")

    # 1. 调用模块1, 加载并预处理数据
    from a1 import NewDataProcessor
    connection_str = "postgresql+psycopg2://postgres:456258@localhost:5432/crypto_data"
    processor = NewDataProcessor(connection_str, freq="30s")


    start_time = "2025-01-20 00:00:00"
    end_time   = "2027-04-04 10:00:00"

    df_merged = processor.get_processed_data(start_time, end_time)
    # ==== 新增：计算对数收益目标列（预测 30 s 之后） ====
    # ==== 计算 Δprice 目标列（预测 30 s 之后） ====
    HORIZON = 1  # 若 freq=10s，则 1×10s=10s，若要 30s 预测则 HORIZON=3
    price = df_merged["kline_close"].astype(float)
    df_merged["dprice"] = price.shift(-HORIZON) - price
    df_merged.dropna(inplace=True)
    # 只做缩放，不中心化
    target_scaler = StandardScaler(with_mean=False)
    df_merged["dprice_scaled"] = target_scaler.fit_transform(df_merged[["dprice"]])
    target_col = "dprice_scaled"

    df_merged.sort_values('time_window', inplace=True)
    if df_merged.empty:
        logging.error("df_merged 为空, 无法继续训练.")
        return

    logging.info(f"成功获取 {len(df_merged)} 条数据, "
                 f"时段: {df_merged['time_window'].min()} ~ {df_merged['time_window'].max()}")

    # =============== 记录本次训练数据的截止时间 ================
    train_end_timestamp = df_merged['time_window'].max()

    # 2. 目标列 & 特征列

    if target_col not in df_merged.columns:
        logging.error(f"目标列 {target_col} 不在数据中, 请修改.")
        return
    input_cols = [c for c in df_merged.columns if c not in ['time_window', target_col]]

    # 3. 构建Dataset (滑动窗口回归)
    seq_len  = 120
    horizon  = 1
    event_cols = []  # 如果有其它事件数据，就填列名
    # -------------------------------------------------------------------
    # 模型超参 (统一管理)
    # -------------------------------------------------------------------
    MODEL_HPARAMS = {
        'proj_dim'        : 128,
        'hidden_dim'      : 384,
        'wave'            : 'db4',
        'level'           : 3,
        'use_gpu_wavelet' : True,
        'event_dim'       : len(event_cols),
        'clamp_val'       : 5.0,
        'spectral'        : False,
        'rtol'            : 1e-2,
        'atol'            : 1e-2,
        'max_step'        : 8,
        'ode_step'        : 0.02,
    }


    # --- 严格按时间切分完  train_df / val_df / test_df  之后 -----------------
    df_merged.sort_values("time_window", inplace=True)  # 先按时间排好
    SEQ_LEN = 120  # 和 Dataset 里的保持一致
    gap = 3 * SEQ_LEN
    total = len(df_merged)
    split1 = int(total * 0.8)
    split2 = int(total * 0.9)

    train_df = df_merged.iloc[:split1 - gap].copy()
    val_df = df_merged.iloc[split1:split2].copy()
    test_df = df_merged.iloc[split2:].copy()



    # ========= 重新构建数据集 & DataLoader  (⇦ 再创建 Dataset) =========
    train_ds = TimeseriesWaveletDataset(train_df, input_cols, target_col,
                                        seq_len=seq_len, horizon=horizon,
                                        event_cols=event_cols)
    val_ds = TimeseriesWaveletDataset(val_df, input_cols, target_col,
                                      seq_len=seq_len, horizon=horizon,
                                      event_cols=event_cols)
    test_ds = TimeseriesWaveletDataset(test_df, input_cols, target_col,
                                       seq_len=seq_len, horizon=horizon,
                                       event_cols=event_cols)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    logging.info(f"Dataset size: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")




    # 4. 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")

    model = HybridNeuralODEModel(
        input_dim=len(input_cols),
        **MODEL_HPARAMS
    ).to(device)

    # --- 加权 Huber + log-cosh 损失 -------------------
    def weighted_loss(pred, target):
        # Huber beta=0.5
        huber_part = nn.SmoothL1Loss(beta=0.5, reduction='none')(pred, target)
        # 自实现 log-cosh
        logcosh_part = log_cosh(pred - target)
        # 给 |y| > std 的大波动样本加倍权重
        weight = 1.0 + (target.abs() > target.std()).float()
        return (huber_part + 0.2 * logcosh_part) * weight

    def criterion(pred, y):
        # 单目标 ΔPrice 回归，乘上配置的权重
        return LOSS_WEIGHT_DELTA_PRICE * weighted_loss(pred, y).mean()

    # -------------------------------------------------

    #use_amp = True
    #scaler = GradScaler(enabled=use_amp)
    base_lr = 1e-3

    optimizer = optim.AdamW(model.parameters(), lr=base_lr,
                            betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    # 初始化 EMA

    ema = EMA(model, decay=0.995)

    warmup_epochs = 10

    def lr_lambda(epoch):
        if epoch < warmup_epochs:  # 0~9 线性升
            return (epoch + 1) / warmup_epochs
        if epoch < 90:  # 10~89 保持 plateau
            return 1.0
        # 90 之后余弦退火 60 轮到 0
        return 0.5 * (1 + np.cos(np.pi * (epoch - 90) / 60))

    # 单周期余弦衰减，避免周期性跳高
    num_epochs = 150
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,

        T_max=num_epochs,
        eta_min=1e-5
    )

    # 6. 训练循环
    #num_epochs = 150
    clip_grad = 5.0
    best_val_loss = float('inf')
    best_state = None
    patience = 10
    wait = 0

    for epoch in range(1, num_epochs+1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,  clip_grad, device, ema)

        val_loss = eval_one_epoch(model, val_loader, criterion, device)
        scheduler.step(epoch-1)

        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Epoch {epoch:02d} – lr={current_lr:.6f}")

        logging.info(f"[Epoch {epoch:02d}] train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        if torch.cuda.is_available():
            # 单位 MB
            curr_alloc = torch.cuda.memory_allocated() / 1024 ** 2
            curr_reserved = torch.cuda.memory_reserved() / 1024 ** 2
            peak_alloc = torch.cuda.max_memory_allocated() / 1024 ** 2
            peak_reserved = torch.cuda.max_memory_reserved() / 1024 ** 2
            logging.info(
                f"GPU Memory | Alloc={curr_alloc:.1f}MB "
                f"| Reserved={curr_reserved:.1f}MB "
                f"| PeakAlloc={peak_alloc:.1f}MB "
                f"| PeakReserved={peak_reserved:.1f}MB"
            )
            # 可选：重置峰值统计，为下一 epoch 做准备
            torch.cuda.reset_peak_memory_stats()

        # Early Stopping + 保存最佳
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

            checkpoint = {
                'epoch': epoch,
                'model_state': best_state,
                'optimizer_state': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'train_end_timestamp': str(train_end_timestamp),
                'input_cols': input_cols,
                'input_dim': len(input_cols),
                'model_config': MODEL_HPARAMS,  # 使用之前定义的 MODEL_HPARAMS
                'target_col_name_at_train': target_col,
                'a1_config': {
                    'freq': '30s',
                    'scaler_path': processor.scaler_path,
                      },

                'ema_shadow_params': ema.shadow_params,

            }

            os.makedirs('artifacts', exist_ok=True)
            torch.save(checkpoint, "checkpoint_best.pth")
            wait = 0


        else:
            wait += 1
            if wait >= patience:
                logging.info(f"验证集连续 {patience} 轮无改善, 提前停止.")
                break

    # 恢复最佳状态
    if best_state:
        # 用 EMA 权重评估／保存最终模型
        ema.apply_shadow(model)

    # 再保存一个仅包含权重的文件
    torch.save(model.state_dict(), "best_model.pth")

    # ========= 7. 测试集 =========
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            x_seq = batch['x_seq'].to(device, non_blocking=True)
            y_val = batch['y_val'].to(device, non_blocking=True)
            evt_seq = batch['event_seq'].to(device, non_blocking=True) if batch['event_seq'].size(-1) > 0 else None

            y_pred = model(x_seq, evt_seq)
            all_pred.append(y_pred.cpu().view(-1, 1).numpy())
            all_true.append(y_val.cpu().view(-1, 1).numpy())

    y_pred_scaled = np.concatenate(all_pred, axis=0)
    y_true_scaled = np.concatenate(all_true, axis=0)

    # —— 反归一化到 USD ——
    y_pred_orig = target_scaler.inverse_transform(y_pred_scaled)
    y_true_orig = target_scaler.inverse_transform(y_true_scaled)

    from sklearn.metrics import mean_squared_error, mean_absolute_error
    mse_orig = mean_squared_error(y_true_orig, y_pred_orig)
    rmse_orig = np.sqrt(mse_orig)
    mae_orig = mean_absolute_error(y_true_orig, y_pred_orig)

    logging.info(
        f"=== 原始尺度 (USD) 测试集指标: RMSE={rmse_orig:.6f}  MAE={mae_orig:.6f}  ==="
    )

    # （可选）把预测结果导出，用于回测或可视化
    #export = test_df.iloc[seq_len + horizon - 1:].copy()
    #export['pred_dprice'] = y_pred_orig.squeeze()
    #export['pred_price'] = export['kline_close'].astype(float) + export['pred_dprice']
    #export.to_csv('latest_price_predictions.csv', index=False)
    #logging.info("预测结果已保存到 latest_price_predictions.csv")


if __name__ == "__main__":
    main()