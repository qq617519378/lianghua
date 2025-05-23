# =======================  config.py  =======================
from pathlib import Path
import torch
import pandas as pd

# ── 损失函数权重 ──────────────────────────────────────────────
LOSS_WEIGHT_LOGRET      = 1.0   # A201 用的对数收益率权重
LOSS_WEIGHT_DELTA_PRICE = 1.0   # A2 用的 ΔPrice 权重，单目标回归一般设为 1.0
# ErrorSweeper 轮询间隔（秒）
ERROR_SWEEPER_INTERVAL_SECONDS = 10

# ── 日志 / 设备 ─────────────────────────────────────────────
APP_NAME   = "TradingSystemV3.Fix4.2.Modular.CN.v1"
LOG_LEVEL  = "INFO"
DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE     = torch.device(DEVICE_STR)
USE_CUDNN_BENCHMARK = True

# ── 数据库连接 ─────────────────────────────────────────────
DB_HOST, DB_PORT, DB_NAME = "localhost", 5432, "crypto_data"
DB_USER, DB_PASS = "postgres", "456258"
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
DB_ENGINE_POOL_SIZE, DB_ENGINE_MAX_OVERFLOW = 5, 10
DB_MARKET_DATA_TABLE, DB_CHAIN_FEATURES_TABLE = "crypto_data", "blockchain_features_utc"
DB_PREDICTION_LOG_TABLE = "prediction_log"

# ── Tick 调度 ─────────────────────────────────────────────
HISTORY_LOOKBACK_DAYS = 30
RAW_CACHE_MAX_HOURS = HISTORY_LOOKBACK_DAYS * 168
MAIN_TICK_INTERVAL_SECONDS = 10
# ── 统一历史数据窗口（天）──
#HISTORY_LOOKBACK_DAYS = 7

# ── A1（30 s）────────────────────────────────────────────
A1_FREQ = "30s"
A1_SCALER_FILENAME = "scaler_30s.joblib"
A1_HORIZON_PERIODS_FOR_A2 = 1
A1_TARGET_COL_RAW_DPRICE_FOR_A2   = "dprice_raw_for_a2_target"
A1_TARGET_COL_SCALED_DPRICE_FOR_A2= "dprice_scaled_for_a2_target"
A1_PROCESSOR_ROLLING_BUFFER_ESTIMATE_STR = "2h 30min"

# ── A101（3 min 特征工程）────────────────────────────────
ARTIFACTS_DIR = Path(r"E:\xinjianwenjianjia\PycharmProjects\PythonProject1\zuixinmoxing\jinping")
A101_AGG_PERIOD      = "3min"
A101_ROLL_WINDOW     = 15
A101_PRED_HORIZON_K  = 1
A101_TARGET_COL_NAME = "future_log_return"
A101_SCALER_FILENAME = f"scaler_{A101_AGG_PERIOD}_rw{A101_ROLL_WINDOW}.joblib"
A101_EXCLUDE_COLS    = ["id", "inserted_at", "height", "mempool_size",
                        "total_mempool_value", "avg_mempool_fee_rate"]
A101_PROCESSOR_ROLLING_BUFFER_ESTIMATE_STR = "240m"
MIN_CHAIN_DATA_HISTORY_FOR_A101_STR = "60d"

# ── A2（Hybrid Neural ODE 价格回归）─────────────────────
A2_SEQ_LEN = 1000
A2_MODEL_CHECKPOINT_PATH = Path("checkpoint_best.pth")
A2_BEST_MODEL_PATH       = Path("best_model.pth")
A2_MODEL_CONSTRUCTOR_PARAMS_DEFAULT = {
    "input_dim": 115,
    "ode_hidden_dim": 256,
    "mlp_hidden_dim": 128,
    "output_dim": 1,
}
SIGNAL_THRESHOLD_DP_A2 = 20.0
ONLINE_LR_A2 = 1e-6

# ── A201（Trend ODE + Wavelet）──────────────────────────
A201_SEQ_LEN = 60
A201_MODEL_CHECKPOINT_PATH = Path(
    "best_ckpt_pytorch_lgbm60_mt_logret_dprice_K1_agg3min_rw15_bktmix.pth"
)
A201_BEST_MODEL_PATH = Path("best_model_a201_ema_weights.pth")
SIGNAL_THRESHOLD_DP_A201 = 50.0
SIGNAL_THRESHOLD_LR_A201 = 0.0003
ONLINE_LR_A201 = 5e-6

# ── 在线学习 / 训练超参 ─────────────────────────────────
#MIN_SAMPLES_FOR_BATCH_LEARNING = ONLINE_BATCH_SIZE
ONLINE_LEARNING_EPOCHS = 1
REPLAY_BUFFER_SIZE     = 1_000
REPLAY_RATIO           = 0.25
EWC_LAMBDA             = 0.01
EMA_DECAY_DEFAULT      = 0.999
ONLINE_BATCH_SIZE = 1000# A2/A201 在线学习每批样本数
MIN_SAMPLES_FOR_BATCH_LEARNING = ONLINE_BATCH_SIZE
MIN_SAMPLES_FOR_ONLINE_LEARNING_TRIGGER = MIN_SAMPLES_FOR_BATCH_LEARNING

# ── 其他 ───────────────────────────────────────────────
try:
    from pytorch_wavelets import DWT1DForward
    HAS_PYTORCH_WAVELETS = True
except ImportError:
    HAS_PYTORCH_WAVELETS = False

VAR = None  # 备用占位
# ================================================================
A201_MODEL_CONSTRUCTOR_PARAMS_DEFAULT = {
    "in_dim": -1,          # 由代码动态覆盖
    "d_model": 192,        # 与训练时保持一致
    "gru_hidden": 256,
    "n_block": 3,
    "gru_layers": 2,
    "n_layer": 3,
    "dropout_transformer": 0.1,
    "dropout_head": 0.1,
    "nhead_transformer": 8,
}
# -------------------------------------------------
# 统一日志中文化
LOG_FORMAT_CN = (
    "%(asctime)s - %(name)s - %(levelname)s - %(threadName)s - %(message)s"
)
MERGE_TOLERANCE_MINUTES = 120      # 行情与链上特征最大允许间隔
# 至少缓存多少分钟的 30 s 行情，供 A2 使用
MIN_HISTORY_MINUTES_A2 = 600        # 4 h，够 128 行 + 训练滚窗
# 离线训练数据最后一条样本时间戳将在加载 checkpoint 时写入
TRAIN_END_TS_A2   = None   # 运行时由 ModelPredictionService 赋值
TRAIN_END_TS_A201 = None

#ONLINE_MIN_DELAY_MINUTES = 300   # 超过该延迟才允许开始在线学习
#ONLINE_BATCH_TRIGGER_SAMPLES = 500

# ffill / bfill 允许用多久的 “旧值” 去填洞
FILL_LIMIT_30S = 120                 # 30 s × 12 = 6 min
# —— 在线学习触发阈值 ——
ONLINE_BATCH_TRIGGER = 5000           # 达到 500 条即可触发
ONLINE_LEARNING_DELAY_MINUTES = 300    # 训练集截止后再过 30 分钟即可触发
# —— 离线训练集的最后时间戳（运行时由 ModelPredictionService 写入） ——
#TRAIN_END_TS_A2   = None        # type: Optional[pd.Timestamp]
#TRAIN_END_TS_A201 = None
ONLINE_LEARNING_INTERVAL_MINUTES = 300          # 两次在线学习最少间隔 5 小时
ONLINE_LEARNING_LOOKBACK_DAYS    = 3            # 为了计算滚动特征往前多取 3 天
ONLINE_MIN_BLOCK_ROWS            = 256          # 如果行数太少就跳过本轮在线学习
ENABLE_ONLINE_LEARNING = False
# —— 抗遗忘 Anchor-Loss 系数 ——
ANCHOR_LAMBDA = 0.02
# ======================= config.py 新增/修改的策略配置 =======================
# ... (你已有的其他配置) ...

STRATEGY_SERVICE_LOGGER_NAME = "交易系统.策略引擎" # 修改日志记录器名称

STRATEGY_ASSET_PAIR = "BTC-USDT-SWAP" # 交易的资产对
STRATEGY_INITIAL_CASH = 10000.0     # 模拟盘初始现金 (USDT)
STRATEGY_FEE_RATE = 0.0005          # 单边手续费率 (例如 0.05%)

STRATEGY_WEIGHT_A2 = 0.6            # A2模型在合并信号时的权重
STRATEGY_WEIGHT_A201 = 0.4          # A201模型在合并信号时的权重
# STRATEGY_TIME_DECAY_TAU_SECONDS = 3600 # 信号时间衰减的tau参数 (秒), 例如1小时 (暂未实现)

STRATEGY_SIGNAL_THRESHOLD_COMBINED = 0.25 # 合并后信号强度低于此值则忽略 (0到1之间)
STRATEGY_TRADE_COOLDOWN_SECONDS = 5.0    # 两笔交易之间的最小冷却时间 (秒)

# --- 头寸规模 ---
STRATEGY_POSITION_SIZING_MODE = "fixed_usdt" # "fixed_usdt", "volatility_adjusted_usdt"
STRATEGY_FIXED_TRADE_SIZE_USDT = 1000.0 # "fixed_usdt"模式下的基础名义价值 (USDT)
STRATEGY_VOL_ADJUSTED_TARGET_RISK_USDT = 50.0 # "volatility_adjusted_usdt"模式下，每次交易目标风险敞口 (USDT)

# --- 风险控制 ---
STRATEGY_MAX_POSITION_UNITS_ASSET = 2.0   # 允许的最大持仓单位 (币本位, 例如 2 BTC)
STRATEGY_MIN_TRADE_UNITS_ASSET = 0.001 # 最小下单单位（币本位，例如 0.001 BTC）
STRATEGY_MAX_DAILY_LOSS_PCT = 0.05    # 最大日亏损百分比 (基于初始资金的5%)
STRATEGY_MAX_CONSECUTIVE_LOSSES = 5   # 最大连续亏损次数 (达到后可能暂停交易)

STRATEGY_ALLOW_FLIPPING = True      # 是否允许在平仓后立即反向开仓
STRATEGY_ALLOW_WEAK_SIGNAL_CLOSE = False # 是否允许弱信号（如WEAK_SELL）仅用于平仓而不开新仓
STRATEGY_MIN_FLIP_INTERVAL_SECONDS = 300.0 # 两次反向开仓之间的最小间隔 (秒)

# --- 止盈止损 ---
STRATEGY_ENABLE_STOP_LOSS = True
STRATEGY_STOP_LOSS_TYPE = "percentage" # "percentage", "atr" (ATR暂未完全集成)
STRATEGY_STOP_LOSS_PERCENTAGE = 0.02   # 基于入场价的2%作为止损
# STRATEGY_STOP_LOSS_ATR_MULTIPLIER = 2.0 # ATR止损的倍数 (暂未实现)

STRATEGY_ENABLE_TAKE_PROFIT = True
STRATEGY_TAKE_PROFIT_TYPE = "percentage" # "percentage", "risk_reward_ratio"
STRATEGY_TAKE_PROFIT_PERCENTAGE = 0.04 # 基于入场价的4%作为止盈
# STRATEGY_TAKE_PROFIT_RR_RATIO = 2.0 # 基于风险回报比的止盈 (例如止损的2倍)
STRATEGY_DYNAMIC_CONFIG_FILE_PATH = ARTIFACTS_DIR / "strategy_params_live.json"
STRATEGY_CONFIG_REFRESH_INTERVAL_SECONDS = 60
# 信号阈值 (这些已经在你的 core_utils.SignalGenerator 中使用了, 确保它们在 config.py 中)
# SIGNAL_THRESHOLD_DP_A2 = 20.0
# SIGNAL_THRESHOLD_DP_A201 = 50.0
# SIGNAL_THRESHOLD_LR_A201 = 0.0003
# =====================================================================

  # "STRATEGY_WEIGHT_A2": 0.6,                            // A2模型在合并信号时的权重 (0.0-1.0)
  # "STRATEGY_WEIGHT_A201": 0.4,                          // A201模型在合并信号时的权重 (0.0-1.0)
  # "STRATEGY_SIGNAL_THRESHOLD_COMBINED": 0.25,           // 合并后信号强度低于此值则忽略 (0.0-1.0)
  # "STRATEGY_TRADE_COOLDOWN_SECONDS": 60,                // 两笔交易之间的最小冷却时间 (秒)
  # "STRATEGY_POSITION_SIZING_MODE": "fixed_usdt",        // 头寸规模模式: "fixed_usdt" 或 "volatility_adjusted_usdt"
  # "STRATEGY_FIXED_TRADE_SIZE_USDT": 1000,               // "fixed_usdt"模式下的基础名义价值 (USDT), 会根据信号强度调整
  # "STRATEGY_VOL_ADJUSTED_TARGET_RISK_USDT": 50,         // "volatility_adjusted_usdt"模式下，每次交易目标风险敞口 (USDT)
  # "STRATEGY_MAX_POSITION_UNITS_ASSET": 2.0,             // 允许的最大持仓单位 (资产本位, 例如 2 BTC)
  # "STRATEGY_MIN_TRADE_UNITS_ASSET": 0.001,              // 最小下单单位 (资产本位, 例如 0.001 BTC)
  # "STRATEGY_MAX_DAILY_LOSS_PCT": 0.05,                  // 最大日亏损百分比 (基于初始资金的5%)
  # "STRATEGY_MAX_CONSECUTIVE_LOSSES": 5,                 // 最大连续亏损次数 (达到后可能暂停交易)
  # "STRATEGY_ALLOW_FLIPPING": true,                      // 是否允许在平仓后立即反向开仓 (true/false)
  # "STRATEGY_ALLOW_WEAK_SIGNAL_CLOSE": false,            // 是否允许弱信号 (如WEAK_SELL) 仅用于平仓而不开新仓 (true/false)
  # "STRATEGY_MIN_FLIP_INTERVAL_SECONDS": 300,            // 两次反向开仓之间的最小间隔 (秒)
  # "STRATEGY_ENABLE_STOP_LOSS": true,                    // 是否启用止损 (true/false)
  # "STRATEGY_STOP_LOSS_TYPE": "percentage",              // 止损类型: "percentage" 或 "atr"
  # "STRATEGY_STOP_LOSS_PERCENTAGE": 0.02,                // "percentage"止损类型的百分比 (例如 0.02 代表2%)
  # "STRATEGY_STOP_LOSS_ATR_MULTIPLIER": 2.0,             // "atr"止损类型的ATR倍数 (例如 ATR值的2倍)
  # "STRATEGY_ENABLE_TAKE_PROFIT": true,                  // 是否启用止盈 (true/false)
  # "STRATEGY_TAKE_PROFIT_TYPE": "percentage",            // 止盈类型: "percentage" 或 "risk_reward_ratio"
  # "STRATEGY_TAKE_PROFIT_PERCENTAGE": 0.04,              // "percentage"止盈类型的百分比 (例如 0.04 代表4%)
  # "STRATEGY_TAKE_PROFIT_RR_RATIO": 2.0,                 // "risk_reward_ratio"止盈类型的风险回报比 (例如止损的2倍)
  # "SIGNAL_THRESHOLD_DP_A2": 20.0,                       // A2模型独立的ΔPrice信号阈值 (USD)
  # "SIGNAL_THRESHOLD_DP_A201": 50.0,                     // A201模型独立的ΔPrice信号阈值 (USD)
  # "SIGNAL_THRESHOLD_LR_A201": 0.0003                    // A201模型独立的对数收益率信号阈值
