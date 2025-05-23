#!/usr/bin/env python
import torch, pandas as pd
from pathlib import Path

from sqlalchemy import create_engine

from data_services import DataPipelineManager
from config import (
    A2_MODEL_CHECKPOINT_PATH, A201_MODEL_CHECKPOINT_PATH,
    DATABASE_URL, DEVICE, A2_SEQ_LEN, A201_SEQ_LEN
)
from a1 import NewDataProcessor
from a101 import preprocess_data as preprocess_a101

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

ckpt_a2 = torch.load(A2_MODEL_CHECKPOINT_PATH, map_location='cpu')
ckpt_a201 = torch.load(A201_MODEL_CHECKPOINT_PATH, map_location='cpu')
cols_a2, cols_a201 = ckpt_a2['input_cols'], ckpt_a201['selected_features_names']
print(f"A2 checkpoint features ({len(cols_a2)}):", cols_a2[:10], "...")
print(f"A201 checkpoint features ({len(cols_a201)}):", cols_a201[:10], "...")

end_dt = pd.Timestamp.utcnow()
start_dt = end_dt - pd.Timedelta(days=30)

dp = NewDataProcessor(db_engine_override=engine, conn_str=None, freq="30s")
df_a1 = dp.get_processed_data(start_dt.strftime('%Y-%m-%d %H:%M:%S'),
                              end_dt.strftime('%Y-%m-%d %H:%M:%S'))

df_a101, _ = preprocess_a101(
    start_dt_str=start_dt.strftime('%Y-%m-%d %H:%M:%S'),
    end_dt_str=end_dt.strftime('%Y-%m-%d %H:%M:%S'),
    pred_horizon_k_lines=1, agg_period="3min", roll_window_override=15,
    db_engine_override=engine
)

print("---- A2 缺失 ----", set(cols_a2) - set(df_a1.columns))
print("---- A201 缺失 ----", set(cols_a201) - set(df_a101.columns))
