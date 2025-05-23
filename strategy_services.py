# --- START OF FILE strategy_services.py ---

# strategy_services.py (策略服务模块 - 全面优化版)

import logging
from pathlib import Path
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List, Union
from datetime import datetime, timezone, timedelta
import numpy as np
import json
import os
import threading
import time
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import warnings
from collections import deque  # 引入 deque

# 导入真实的 config 模块和 SignalGenerator 类
import config
from core_utils import SignalGenerator

logger = logging.getLogger(config.STRATEGY_SERVICE_LOGGER_NAME)

# --- 提取的常量 ---
FLOAT_COMPARISON_TOLERANCE = 1e-9
EQUITY_CURVE_UPDATE_INTERVAL_SECONDS = 30
EQUITY_CURVE_MAX_LENGTH = 5000
STRATEGY_LOOP_TIMEOUT_SECONDS = 60
HEARTBEAT_TIMEOUT_SECONDS = 300
RECENT_SIGNALS_TTL_HOURS = 1


# --- LOGICAL FILE: strategy_enums.py (or similar) ---
class StrategyState(Enum):
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class TradingMode(Enum):
    LIVE = "LIVE"
    PAPER = "PAPER"
    BACKTEST = "BACKTEST"


# --- LOGICAL FILE: strategy_config_manager.py ---
class StrategyConfigManager:
    def __init__(self, default_config_module=config):
        self._lock = threading.RLock()
        self.config_data: Dict[str, Any] = {}
        self.default_config_module = default_config_module
        self.dynamic_config_file_path = Path(config.STRATEGY_DYNAMIC_CONFIG_FILE_PATH)
        self._last_modified_time = 0.0
        self._last_config_snapshot_for_comparison: Dict[str, Any] = {}
        self._file_watcher_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()

        self._validation_rules = {
            # 权重类 - 明确为 float
            'STRATEGY_WEIGHT_A2': {'type': float, 'range': (0.0, 1.0), 'required': True},
            'STRATEGY_WEIGHT_A201': {'type': float, 'range': (0.0, 1.0), 'required': True},
            'STRATEGY_DEFAULT_WEIGHT_OTHER_MODELS': {'type': float, 'range': (0.0, 1.0), 'required': False},

            # 阈值类 (百分比或比率) - 明确为 float
            'STRATEGY_SIGNAL_THRESHOLD_COMBINED': {'type': float, 'range': (0.0, 1.0), 'required': True},
            'STRATEGY_MAX_DAILY_LOSS_PCT': {'type': float, 'range': (0.001, 0.5), 'required': True},
            'STRATEGY_STOP_LOSS_PERCENTAGE': {'type': float, 'range': (0.001, 0.2), 'required': False},
            'STRATEGY_TAKE_PROFIT_PERCENTAGE': {'type': float, 'range': (0.001, 1.0), 'required': False},
            'STRATEGY_STOP_LOSS_ATR_MULTIPLIER': {'type': float, 'range': (0.1, 10.0), 'required': False},
            'STRATEGY_TAKE_PROFIT_RR_RATIO': {'type': float, 'range': (0.5, 10.0), 'required': False},
            'STRATEGY_MAX_DRAWDOWN_PCT': {'type': float, 'range': (0.001, 0.5), 'required': False},
            'STRATEGY_MIN_CONFIDENCE_FOR_WEIGHTING': {'type': float, 'range': (0.0, 1.0), 'required': False},
            'STRATEGY_LOGRET_TO_STRENGTH_DIVISOR': {'type': float, 'range': (0.0001, 1.0), 'required': False},
            'STRATEGY_URGENCY_THRESHOLD_HIGH': {'type': float, 'range': (0.0, 1.0), 'required': False},
            'STRATEGY_URGENCY_THRESHOLD_CRITICAL': {'type': float, 'range': (0.0, 1.0), 'required': False},
            'STRATEGY_FEE_RATE': {'type': float, 'range': (0.0, 0.1), 'required': True},

            # 金额或数量类 (可以是整数或浮点数，但通常用浮点数更灵活) - 明确为 float
            'STRATEGY_FIXED_TRADE_SIZE_USDT': {'type': float, 'range': (1.0, 100000.0), 'required': True},
            'STRATEGY_MAX_POSITION_UNITS_ASSET': {'type': float, 'range': (0.001, 1000.0), 'required': True},
            'STRATEGY_MIN_TRADE_UNITS_ASSET': {'type': float, 'range': (0.00001, 10.0), 'required': True},
            'STRATEGY_INITIAL_CASH': {'type': float, 'range': (0.0, float('inf')), 'required': True},
            'STRATEGY_DYNAMIC_UNITS_MAX_FACTOR': {'type': float, 'range': (1.0, 100.0), 'required': False},

            # 时间间隔类 (可以是整数或浮点数秒) - 保持 (int, float) 或明确为 float，这里选择 float 更通用
            'STRATEGY_TRADE_COOLDOWN_SECONDS': {'type': float, 'range': (0.0, 3600.0), 'required': True},
            'STRATEGY_MIN_FLIP_INTERVAL_SECONDS': {'type': float, 'range': (0.0, 3600.0), 'required': True},
            'STRATEGY_MAIN_LOOP_INTERVAL_SECONDS': {'type': float, 'range': (1.0, 300.0), 'required': False},
            'STRATEGY_HEARTBEAT_TIMEOUT_SECONDS': {'type': float, 'range': (10.0, 3600.0), 'required': False},

            # 计数或固定整数类 - 保持 int
            'STRATEGY_MAX_CONSECUTIVE_LOSSES': {'type': int, 'range': (1, 50), 'required': True},
            'STRATEGY_MIN_MODELS_FOR_SIGNAL': {'type': int, 'range': (1, 10), 'required': False},
            'STRATEGY_MAX_SIGNALS_PER_EXECUTION': {'type': int, 'range': (1, 100), 'required': False},
            'STRATEGY_PREDICTION_CACHE_MAXLEN': {'type': int, 'range': (10, 1000), 'required': False},
            'STRATEGY_CONFIG_REFRESH_INTERVAL_SECONDS': {'type': int, 'range': (5, 3600), 'required': True},  # 这个通常是整数秒

            # 布尔类 - 保持 bool
            'STRATEGY_USE_CONFIDENCE_WEIGHTING': {'type': bool, 'required': False},
            'STRATEGY_USE_FIXED_TRADE_SIZE_USDT': {'type': bool, 'required': False},
            # (config.py 中定义的 STRATEGY_ALLOW_FLIPPING, STRATEGY_ENABLE_STOP_LOSS 等也应该是 bool)
            # 如果它们也需要动态配置，也要加入这里并设 type: bool

            # 字符串类 - 保持 str
            'STRATEGY_ASSET_PAIR': {'type': str, 'required': True},
            # (config.py 中定义的 STRATEGY_POSITION_SIZING_MODE, STRATEGY_STOP_LOSS_TYPE 等也应该是 str)
        }
        self.load_initial_config()
        logger.info("策略配置管理器已初始化")

    def __enter__(self):
        self.start_file_watcher()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def load_initial_config(self):
        with self._lock:
            logger.info("开始加载初始策略配置...")
            self.config_data.clear()  # 清空旧数据，确保从干净状态加载
            for attr_name in dir(self.default_config_module):
                if attr_name.startswith("STRATEGY_"):
                    value = getattr(self.default_config_module, attr_name)
                    self.config_data[attr_name] = value
            logger.info(
                f"已从默认配置模块 '{self.default_config_module.__name__}' 加载{len(self.config_data)}个初始策略参数")

            try:
                self._validate_config()
            except ValueError as e:
                logger.critical(f"默认配置 (config.py) 未通过验证: {e}. 请修正 config.py。")
                raise  # 阻止服务启动，如果基础配置就有问题

            self._load_from_dynamic_file(is_initial_load=True)
            self._last_config_snapshot_for_comparison = self.config_data.copy()

    def _validate_config(self):
        errors = []
        for param_name, rules in self._validation_rules.items():
            is_present_in_config_data = param_name in self.config_data

            if rules.get('required', False) and not is_present_in_config_data:
                errors.append(f"缺少必需配置: {param_name}")
                continue

            if is_present_in_config_data:
                value = self.config_data[param_name]
                expected_types = rules.get('type')
                if expected_types and not isinstance(value, expected_types):
                    if not (expected_types is bool and isinstance(value, int) and value in (0, 1)):
                        errors.append(
                            f"配置 {param_name} 类型错误: 期望 {expected_types}, 实际 {type(value)} (值: {value!r})")

                value_range = rules.get('range')
                if value_range and isinstance(value, (int, float)):
                    if not (value_range[0] <= value <= value_range[1]):
                        errors.append(f"配置 {param_name} 值 {value!r} 超出范围: [{value_range[0]}, {value_range[1]}]")

        val_a2 = self.config_data.get('STRATEGY_WEIGHT_A2')
        val_a201 = self.config_data.get('STRATEGY_WEIGHT_A201')
        if isinstance(val_a2, (int, float)) and isinstance(val_a201, (int, float)):
            weight_sum = val_a2 + val_a201
            if abs(weight_sum - 1.0) > FLOAT_COMPARISON_TOLERANCE:
                errors.append(
                    f"权重之和不等于1: A2={val_a2}, A201={val_a201}, Sum={weight_sum}")
        elif val_a2 is not None or val_a201 is not None:  # 如果至少一个存在但类型不对
            errors.append(f"权重 STRATEGY_WEIGHT_A2 或 STRATEGY_WEIGHT_A201 类型非数值或缺失，无法检查总和。")

        min_trade = self.config_data.get('STRATEGY_MIN_TRADE_UNITS_ASSET')
        max_pos = self.config_data.get('STRATEGY_MAX_POSITION_UNITS_ASSET')
        if min_trade is not None and max_pos is not None:
            if isinstance(min_trade, (int, float)) and isinstance(max_pos, (int, float)):
                if min_trade >= max_pos:
                    errors.append(f"最小交易单位 ({min_trade}) 不能大于等于最大持仓单位 ({max_pos})")
            else:
                errors.append(
                    f"STRATEGY_MIN_TRADE_UNITS_ASSET 或 STRATEGY_MAX_POSITION_UNITS_ASSET 类型非数值，无法比较。")

        if errors:
            error_msg = "配置验证失败:\n" + "\n".join(errors)
            # logger.error(error_msg) # _load_from_dynamic_file 会记录，这里只抛出
            raise ValueError(error_msg)

    def _validate_and_apply_dynamic_param(self, key: str, new_value: Any) -> bool:
        if key not in self.config_data and key not in self._validation_rules:
            # logger.warning(f"动态配置'{key}'在初始配置和验证规则中均不存在，已忽略。") # 可能过于冗余
            return False

        current_value = self.config_data.get(key)
        rules_for_key = self._validation_rules.get(key, {})
        expected_types_from_rules = rules_for_key.get('type')

        target_type_for_conversion: Optional[Union[type, Tuple[type, ...]]] = None
        if expected_types_from_rules:
            target_type_for_conversion = expected_types_from_rules[0] if isinstance(expected_types_from_rules,
                                                                                    tuple) else expected_types_from_rules
        elif current_value is not None:
            target_type_for_conversion = type(current_value)

        try:
            converted_value = self._convert_value_type(new_value, target_type_for_conversion)

            if key in self._validation_rules:
                rules = self._validation_rules[key]
                expected_types_rule = rules.get('type')
                if expected_types_rule and not isinstance(converted_value, expected_types_rule):
                    if not (expected_types_rule is bool and isinstance(converted_value, int) and converted_value in (
                    0, 1)):
                        logger.warning(
                            f"动态配置'{key}'类型不匹配 ({type(converted_value)} vs {expected_types_rule})，已忽略")
                        return False

                value_range = rules.get('range')
                if value_range and isinstance(converted_value, (int, float)):
                    if not (value_range[0] <= converted_value <= value_range[1]):
                        logger.warning(
                            f"动态配置'{key}'值 {converted_value!r} 超出范围 [{value_range[0]}, {value_range[1]}]，已忽略")
                        return False

            is_changed = False
            if current_value is None and converted_value is not None:
                is_changed = True
            elif current_value is not None and converted_value is None:
                is_changed = True
            elif isinstance(current_value, float) and isinstance(converted_value, float):
                is_changed = not np.isclose(current_value, converted_value, atol=FLOAT_COMPARISON_TOLERANCE)
            else:
                is_changed = current_value != converted_value

            if is_changed:
                old_value_display = self.config_data.get(key, "N/A (new key)")
                self.config_data[key] = converted_value
                logger.info(f"动态配置'{key}'已更新: {old_value_display!r} -> {converted_value!r}")
                return True
            return False

        except ValueError as e:
            logger.error(f"转换动态配置'{key}'(值: {new_value!r}, 目标类型: {target_type_for_conversion})时出错: {e}")
            return False
        except Exception as e:
            logger.error(f"验证或应用动态配置'{key}'(值: {new_value!r})时发生未知错误: {e}", exc_info=True)
            return False

    def _convert_value_type(self, value: Any, expected_type: Optional[Union[type, Tuple[type, ...]]]) -> Any:
        actual_expected_type: Optional[type] = None
        if isinstance(expected_type, tuple):
            actual_expected_type = expected_type[0]
            if any(isinstance(value, t) for t in expected_type):
                return value
        else:
            actual_expected_type = expected_type

        if actual_expected_type is None or isinstance(value, actual_expected_type):
            return value

        try:
            if actual_expected_type is float: return float(value)
            if actual_expected_type is int: return int(float(value))
            if actual_expected_type is bool:
                if isinstance(value, str):
                    return value.lower() in ['true', '1', 'yes', 'on', 't']
                return bool(int(value)) if isinstance(value, (int, float)) else bool(value)
            if actual_expected_type is Path: return Path(value)
            if actual_expected_type is str: return str(value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"无法将值 '{value!r}' (类型 {type(value)}) 转换为 {actual_expected_type}: {e}")

        raise ValueError(f"不支持的类型转换：从 {type(value)} 到 {actual_expected_type}")

    def _load_from_dynamic_file(self, is_initial_load: bool = False):
        try:
            if not self.dynamic_config_file_path.exists():
                if is_initial_load:
                    logger.info(f"动态配置文件'{self.dynamic_config_file_path}'未找到。将仅使用默认配置。")
                return

            current_modified_time = os.path.getmtime(self.dynamic_config_file_path)
            if not is_initial_load and current_modified_time <= self._last_modified_time:
                return

            with open(self.dynamic_config_file_path, 'r', encoding='utf-8') as f:
                dynamic_params = json.load(f)
                logger.critical(f"DEBUG _load_from_dynamic_file: Parsed dynamic_params = {dynamic_params}")

            updated_keys = []
            critical_changes = []

            with self._lock:
                config_before_dynamic_load = self.config_data.copy()

                for key, value in dynamic_params.items():
                    if key.startswith("STRATEGY_"):
                        if self._validate_and_apply_dynamic_param(key, value):
                            updated_keys.append(key)
                            if key in ['STRATEGY_ASSET_PAIR', 'STRATEGY_INITIAL_CASH',
                                       'STRATEGY_MAX_POSITION_UNITS_ASSET']:
                                critical_changes.append(key)
                try:
                    self._validate_config()
                except ValueError as e:
                    logger.error(f"动态配置应用后，整体配置校验失败: {e}. "
                                 f"配置可能处于不一致状态。正在尝试回滚到加载前状态...")
                    self.config_data = config_before_dynamic_load
                    try:
                        self._validate_config()
                        logger.info("配置已成功回滚到动态加载前的状态。")
                    except ValueError as ve_rollback:
                        logger.critical(f"回滚配置后校验仍失败: {ve_rollback}. 配置管理器可能处于严重错误状态！")
                    updated_keys = []
                    critical_changes = []
                    return

                self._last_modified_time = current_modified_time
                if updated_keys:
                    self._last_config_snapshot_for_comparison = self.config_data.copy()

            if updated_keys:
                if critical_changes:
                    logger.critical(f"动态配置更新了关键参数: {critical_changes}。请检查策略行为是否符合预期。")
                logger.info(f"从动态文件成功更新了 {len(updated_keys)} 个策略参数: {updated_keys}")
            elif not is_initial_load:
                logger.debug("动态配置文件已检查，无有效或已验证的参数变更。")

        except json.JSONDecodeError as e:
            logger.error(f"解析动态配置文件 '{self.dynamic_config_file_path}' 失败 (JSON格式错误): {e}")
        except Exception as e:
            logger.error(f"加载动态配置文件时发生未知错误: {e}", exc_info=True)

    def start_file_watcher(self):
        if self._file_watcher_thread is not None and self._file_watcher_thread.is_alive():
            # logger.debug("文件监控线程已在运行中。") # 过于频繁的日志
            return

        refresh_interval = getattr(self.default_config_module, 'STRATEGY_CONFIG_REFRESH_INTERVAL_SECONDS', 60)

        def watcher_thread():
            logger.info(
                f"配置文件监控线程已启动，监控文件: {self.dynamic_config_file_path}, 刷新间隔: {refresh_interval}s")
            while not self._shutdown_event.is_set():
                try:
                    self._load_from_dynamic_file()
                except Exception as e:
                    logger.error(f"文件监控线程在调用 _load_from_dynamic_file 时发生错误: {e}", exc_info=True)
                if self._shutdown_event.wait(refresh_interval):
                    break
            logger.info("配置文件监控线程已停止")

        self._file_watcher_thread = threading.Thread(
            target=watcher_thread, daemon=True, name="StrategyConfigWatcher"
        )
        self._file_watcher_thread.start()

    def shutdown(self):
        logger.info("正在关闭配置管理器...")
        self._shutdown_event.set()
        if self._file_watcher_thread and self._file_watcher_thread.is_alive():
            self._file_watcher_thread.join(timeout=5.0)
            if self._file_watcher_thread.is_alive():
                logger.warning("文件监控线程未能在超时内停止。")
        logger.info("配置管理器已关闭。")

    def get(self, param_name: str, default_value: Optional[Any] = None) -> Any:
        with self._lock:
            if param_name in self.config_data:
                return self.config_data[param_name]
        if hasattr(self.default_config_module, param_name):
            return getattr(self.default_config_module, param_name)
        return default_value

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            # 正确处理方式是让 Python 自己处理或调用 super
            try:
                return self.__getattribute__(name)  # 尝试正常属性访问
            except AttributeError:
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        if name.startswith("STRATEGY_"):
            with self._lock:
                if name in self.config_data:
                    return self.config_data[name]
            if hasattr(self.default_config_module, name):
                return getattr(self.default_config_module, name)
        raise AttributeError(f"'{type(self).__name__}' 对象或其默认配置模块没有属性 '{name}'")


# --- LOGICAL FILE: strategy_data_structures.py ---
@dataclass(frozen=True)
class PredictionResult:
    model_name: str
    ts_pred: pd.Timestamp
    horizon_sec: int
    base_price: float
    pred_logret: Optional[float] = None
    pred_dprice: Optional[float] = None
    feature_hash: Optional[str] = None
    model_version: Optional[str] = None
    atr_at_pred_time: Optional[float] = None
    confidence: Optional[float] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.pred_logret is None and self.pred_dprice is None:
            raise ValueError("pred_logret和pred_dprice不能同时为None")
        if not isinstance(self.base_price, (int, float)) or self.base_price <= 0:
            raise ValueError(f"base_price必须为正数，得到: {self.base_price}")
        if not isinstance(self.horizon_sec, int) or self.horizon_sec <= 0:
            raise ValueError(f"horizon_sec必须为正整数，得到: {self.horizon_sec}")
        if self.confidence is not None and not (0 <= self.confidence <= 1):
            warnings.warn(f"Prediction confidence {self.confidence} for model {self.model_name} "
                          f"is outside the recommended [0, 1] range.")

    @property
    def target_price_from_dprice(self) -> Optional[float]:
        return self.base_price + self.pred_dprice if self.pred_dprice is not None else None

    @property
    def target_price_from_logret(self) -> Optional[float]:
        return self.base_price * np.exp(self.pred_logret) if self.pred_logret is not None else None

    @property
    def ts_target(self) -> pd.Timestamp:
        return self.ts_pred + pd.Timedelta(seconds=self.horizon_sec)

    @property
    def is_expired(self) -> bool:
        return pd.Timestamp.now(tz=timezone.utc) > self.ts_target

    @property
    def time_to_expiry(self) -> float:
        remaining = (self.ts_target - pd.Timestamp.now(tz=timezone.utc)).total_seconds()
        return max(0.0, remaining)


@dataclass(frozen=True)
class TradeSignal:
    timestamp: pd.Timestamp
    source_model_names: List[str]
    asset_pair: str
    signal_type: str
    strength: float
    execution_units: float = 0.0
    target_price_entry: Optional[float] = None
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    reason: str = ""
    raw_predictions_details: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    urgency: str = "NORMAL"

    def __post_init__(self):
        if not (0 <= self.strength <= 1):
            raise ValueError(f"信号强度 {self.strength} 必须在[0, 1]范围内")
        if self.signal_type not in ["BUY", "SELL"]:
            raise ValueError(f"无效的 signal_type: {self.signal_type}")
        if self.urgency not in ["NORMAL", "HIGH", "CRITICAL"]:
            raise ValueError(f"urgency '{self.urgency}' 必须是NORMAL, HIGH或CRITICAL")
        if abs(self.execution_units) < FLOAT_COMPARISON_TOLERANCE and self.signal_type:
            pass  # 允许0单位信号，例如用于“保持”或“无操作”的明确信号


# --- LOGICAL FILE: portfolio_state.py ---
@dataclass
class PortfolioState:
    asset_pair: str
    config_manager: StrategyConfigManager
    current_position_units: float = 0.0
    average_entry_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl_session: float = 0.0
    initial_cash: float = field(init=False)
    cash_balance: float = field(init=False)
    last_trade_timestamp: Optional[pd.Timestamp] = None
    equity_curve: deque = field(default_factory=lambda: deque(maxlen=EQUITY_CURVE_MAX_LENGTH))
    num_trades_session: int = 0
    num_wins_session: int = 0
    num_losses_session: int = 0
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    last_flip_timestamp: Optional[pd.Timestamp] = None
    max_unrealized_profit_session: float = 0.0
    min_unrealized_loss_session: float = 0.0
    total_fees_paid: float = 0.0

    def __post_init__(self):
        self.initial_cash = float(self.config_manager.STRATEGY_INITIAL_CASH)
        self.cash_balance = self.initial_cash
        if not self.equity_curve:
            self.equity_curve.append((pd.Timestamp.now(tz=timezone.utc), self.initial_cash))

    def _round_decimal(self, value: Union[float, str, Decimal], precision: int = 8) -> float:
        if not isinstance(value, (int, float, Decimal, str)):
            raise TypeError(f"Value for rounding must be numeric or string, got {type(value)}")
        try:
            return float(Decimal(str(value)).quantize(Decimal('1e-' + str(precision)), rounding=ROUND_HALF_UP))
        except Exception as e:
            # logger.error(f"Rounding error for value '{value}' (precision {precision}): {e}") # PortfolioState可能没有logger
            warnings.warn(
                f"Rounding error for value '{value}' (precision {precision}): {e}. Returning unrounded float.")
            return float(value)

    def _calculate_trade_pnl_and_update_stats(self, units_closed_abs: float,
                                              original_position_was_long: bool,
                                              entry_price: float, exit_price: float):
        pnl: float
        if original_position_was_long:
            pnl = (exit_price - entry_price) * units_closed_abs
        else:
            pnl = (entry_price - exit_price) * units_closed_abs

        pnl = self._round_decimal(pnl)
        self.realized_pnl_session += pnl

        if pnl > FLOAT_COMPARISON_TOLERANCE:
            self.num_wins_session += 1;
            self.consecutive_losses = 0;
            self.consecutive_wins += 1
        elif pnl < -FLOAT_COMPARISON_TOLERANCE:
            self.num_losses_session += 1;
            self.consecutive_wins = 0;
            self.consecutive_losses += 1

        logger.info(f"平仓交易 PnL: {pnl:.4f} (单位: {units_closed_abs:.6f}, "
                    f"方向: {'Long' if original_position_was_long else 'Short'}, "
                    f"入场: {entry_price:.2f}, 出场: {exit_price:.2f}). "
                    f"会话已实现PnL: {self.realized_pnl_session:.4f}")

    def update_position(self, trade_units: float, trade_price: float):
        if abs(trade_units) < FLOAT_COMPARISON_TOLERANCE:
            logger.debug("交易单位接近零，跳过持仓更新")
            return
        if trade_price <= 0:
            logger.error(f"无效的交易价格: {trade_price}, 跳过持仓更新。")
            return

        fee_rate = self.config_manager.STRATEGY_FEE_RATE
        self.num_trades_session += 1
        trade_value_gross = abs(trade_units) * trade_price
        fees = self._round_decimal(trade_value_gross * fee_rate)
        self.total_fees_paid += fees
        cash_flow_from_trade = -self._round_decimal(trade_units * trade_price)
        current_pos_before_trade = self.current_position_units
        avg_entry_before_trade = self.average_entry_price
        original_position_was_long = current_pos_before_trade > FLOAT_COMPARISON_TOLERANCE
        original_position_was_short = current_pos_before_trade < -FLOAT_COMPARISON_TOLERANCE

        is_closing_or_reducing = False
        if avg_entry_before_trade is not None:
            if original_position_was_long and trade_units < -FLOAT_COMPARISON_TOLERANCE:
                is_closing_or_reducing = True
            elif original_position_was_short and trade_units > FLOAT_COMPARISON_TOLERANCE:
                is_closing_or_reducing = True

        if is_closing_or_reducing:
            units_effectively_closed_abs = min(abs(trade_units), abs(current_pos_before_trade))
            self._calculate_trade_pnl_and_update_stats(
                units_effectively_closed_abs, original_position_was_long,
                avg_entry_before_trade, trade_price  # type: ignore
            )

        new_position_units_raw = current_pos_before_trade + trade_units
        new_position_units = self._round_decimal(new_position_units_raw, 8)

        if abs(new_position_units) < FLOAT_COMPARISON_TOLERANCE:
            self.current_position_units = 0.0;
            self.average_entry_price = None
            self.max_unrealized_profit_session = 0.0;
            self.min_unrealized_loss_session = 0.0
        elif (abs(current_pos_before_trade) < FLOAT_COMPARISON_TOLERANCE) or \
                (current_pos_before_trade * new_position_units < -FLOAT_COMPARISON_TOLERANCE):
            self.current_position_units = new_position_units;
            self.average_entry_price = trade_price
            self.max_unrealized_profit_session = 0.0;
            self.min_unrealized_loss_session = 0.0
            if abs(current_pos_before_trade) > FLOAT_COMPARISON_TOLERANCE:
                self.last_flip_timestamp = pd.Timestamp.now(tz=timezone.utc)
                logger.info(f"检测到反手交易: 从 {current_pos_before_trade:.6f} 到 {new_position_units:.6f}")
        else:
            if avg_entry_before_trade is None:
                logger.error("持仓更新逻辑错误：同向加仓但前均价为空。将使用当前交易价格作为均价。")
                self.average_entry_price = trade_price
            else:
                total_value_old = avg_entry_before_trade * current_pos_before_trade
                trade_value_current = trade_price * trade_units
                if abs(new_position_units) < FLOAT_COMPARISON_TOLERANCE:
                    logger.error("同向加仓计算新均价时，新仓位接近零，逻辑错误。")
                    self.average_entry_price = trade_price
                else:
                    self.average_entry_price = self._round_decimal(
                        (total_value_old + trade_value_current) / new_position_units
                    )
            self.current_position_units = new_position_units

        self.cash_balance = self._round_decimal(self.cash_balance + cash_flow_from_trade - fees)
        self.last_trade_timestamp = pd.Timestamp.now(tz=timezone.utc)
        self.update_equity_curve()
        avg_price_display = f"{self.average_entry_price:.2f}" if self.average_entry_price is not None else "N/A"
        logger.info(f"持仓更新: 交易 {trade_units:.6f} @{trade_price:.2f} "
                    f"(Fee {fees:.4f}). 新仓位: {self.current_position_units:.6f}, "
                    f"均价: {avg_price_display}, 现金: {self.cash_balance:.2f}")

    def update_unrealized_pnl(self, current_market_price: float):
        if abs(self.current_position_units) < FLOAT_COMPARISON_TOLERANCE or self.average_entry_price is None:
            self.unrealized_pnl = 0.0
            self.max_unreal_profit_session = 0.0;
            self.min_unrealized_loss_session = 0.0  # Typo fixed
        else:
            if self.current_position_units > 0:
                pnl = (current_market_price - self.average_entry_price) * self.current_position_units
            else:
                pnl = (self.average_entry_price - current_market_price) * abs(self.current_position_units)
            self.unrealized_pnl = self._round_decimal(pnl)
            self.max_unrealized_profit_session = max(self.max_unrealized_profit_session, self.unrealized_pnl)
            self.min_unrealized_loss_session = min(self.min_unrealized_loss_session, self.unrealized_pnl)
        self.update_equity_curve()

    def update_equity_curve(self):
        current_equity = self._round_decimal(self.cash_balance + self.unrealized_pnl)
        now = pd.Timestamp.now(tz=timezone.utc)
        if self.equity_curve:
            last_time, last_equity = self.equity_curve[-1]
            time_elapsed = (now - last_time).total_seconds()
            equity_changed_significantly = abs(current_equity - last_equity) > (self.initial_cash * 0.0001)
            if time_elapsed < EQUITY_CURVE_UPDATE_INTERVAL_SECONDS and not equity_changed_significantly:
                return
        self.equity_curve.append((now, current_equity))

    @property
    def current_equity(self) -> float:
        return self._round_decimal(self.cash_balance + self.unrealized_pnl)

    @property
    def total_return_pct(self) -> float:
        if self.initial_cash > FLOAT_COMPARISON_TOLERANCE:
            return (self.current_equity - self.initial_cash) / self.initial_cash * 100
        return 0.0

    @property
    def win_rate(self) -> float:
        total_closed_trades = self.num_wins_session + self.num_losses_session
        if total_closed_trades > 0:
            return self.num_wins_session / total_closed_trades * 100
        return 0.0

    def get_performance_summary(self) -> Dict[str, Any]:
        return {
            'asset_pair': self.asset_pair, 'initial_cash': self.initial_cash, 'cash_balance': self.cash_balance,
            'current_position_units': self.current_position_units, 'average_entry_price': self.average_entry_price,
            'unrealized_pnl': self.unrealized_pnl, 'realized_pnl_session': self.realized_pnl_session,
            'current_equity': self.current_equity, 'total_return_pct': self.total_return_pct,
            'num_trades_session': self.num_trades_session, 'num_wins_session': self.num_wins_session,
            'num_losses_session': self.num_losses_session, 'win_rate': self.win_rate,
            'consecutive_wins': self.consecutive_wins, 'consecutive_losses': self.consecutive_losses,
            'max_unrealized_profit_current_pos': self.max_unrealized_profit_session,
            'min_unrealized_loss_current_pos': self.min_unrealized_loss_session,  # Represents max loss magnitude
            'total_fees_paid': self.total_fees_paid,
            'last_trade_timestamp': self.last_trade_timestamp.isoformat() if self.last_trade_timestamp else None,
            'last_flip_timestamp': self.last_flip_timestamp.isoformat() if self.last_flip_timestamp else None,
        }


# --- LOGICAL FILE: risk_manager.py ---
class RiskManager:
    def __init__(self, portfolio_state_ref: PortfolioState, config_manager_ref: StrategyConfigManager):
        self.portfolio: PortfolioState = portfolio_state_ref
        self.config_manager: StrategyConfigManager = config_manager_ref
        self.is_trading_halted_daily_loss: bool = False
        self.is_trading_halted_consecutive_loss: bool = False
        self.is_trading_halted_drawdown: bool = False
        self.daily_peak_equity: float = self.portfolio.initial_cash
        logger.info("风险管理器已初始化")
        self.reset_daily_flags()

    def _log_current_limits(self):
        try:
            max_pos = self.config_manager.STRATEGY_MAX_POSITION_UNITS_ASSET
            min_trade = self.config_manager.STRATEGY_MIN_TRADE_UNITS_ASSET
            daily_loss_pct_limit = self.config_manager.STRATEGY_MAX_DAILY_LOSS_PCT * 100
            con_loss_limit = self.config_manager.STRATEGY_MAX_CONSECUTIVE_LOSSES
            drawdown_pct_limit_val = self.config_manager.get('STRATEGY_MAX_DRAWDOWN_PCT')
            drawdown_display = f"{drawdown_pct_limit_val * 100:.1f}%" if drawdown_pct_limit_val is not None else "N/A (未配置)"
            logger.info(
                f"当前风控参数: MaxPos={max_pos}, MinTrade={min_trade}, "
                f"DailyLossLimit={daily_loss_pct_limit:.1f}%, ConLossLimit={con_loss_limit}, "
                f"MaxDrawdownLimit={drawdown_display}"
            )
        except AttributeError as e:
            logger.error(f"记录风控参数时配置项缺失: {e}. 检查 config.py。")
        except Exception as e:
            logger.error(f"记录风控参数时发生未知错误: {e}", exc_info=True)

    def check_daily_loss_limit(self) -> bool:
        if self.portfolio.initial_cash <= FLOAT_COMPARISON_TOLERANCE: return True
        current_session_loss = -self.portfolio.realized_pnl_session if self.portfolio.realized_pnl_session < 0 else 0.0
        if current_session_loss <= 0: return True
        daily_loss_pct = current_session_loss / self.portfolio.initial_cash
        max_daily_loss_pct = self.config_manager.STRATEGY_MAX_DAILY_LOSS_PCT
        if daily_loss_pct >= max_daily_loss_pct - FLOAT_COMPARISON_TOLERANCE:
            if not self.is_trading_halted_daily_loss:
                self.is_trading_halted_daily_loss = True
                logger.critical(
                    f"触发日亏损限制! 会话亏损 {daily_loss_pct * 100:.2f}% >= 限制 {max_daily_loss_pct * 100:.2f}%. 暂停交易。")
            return False
        return True

    def check_consecutive_loss_limit(self) -> bool:
        max_consecutive_losses = self.config_manager.STRATEGY_MAX_CONSECUTIVE_LOSSES
        if self.portfolio.consecutive_losses >= max_consecutive_losses:
            if not self.is_trading_halted_consecutive_loss:
                self.is_trading_halted_consecutive_loss = True
                logger.critical(
                    f"触发连续亏损限制! 连亏 {self.portfolio.consecutive_losses} >= 限制 {max_consecutive_losses}. 暂停交易。")
            return False
        return True

    def check_drawdown_limit(self) -> bool:
        max_drawdown_pct_config = self.config_manager.get('STRATEGY_MAX_DRAWDOWN_PCT')
        if max_drawdown_pct_config is None: return True
        current_equity = self.portfolio.current_equity
        if current_equity > self.daily_peak_equity: self.daily_peak_equity = current_equity
        if self.daily_peak_equity <= FLOAT_COMPARISON_TOLERANCE: return True
        drawdown = self.daily_peak_equity - current_equity
        if drawdown <= 0: return True
        drawdown_pct = drawdown / self.daily_peak_equity
        if drawdown_pct >= max_drawdown_pct_config - FLOAT_COMPARISON_TOLERANCE:
            if not self.is_trading_halted_drawdown:
                self.is_trading_halted_drawdown = True
                logger.critical(
                    f"触发最大回撤限制! 回撤 {drawdown_pct * 100:.2f}% >= 限制 {max_drawdown_pct_config * 100:.2f}%. 暂停交易。")
            return False
        return True

    def validate_trade_size(self, proposed_units: float) -> Tuple[bool, float, str]:
        abs_proposed_units = abs(proposed_units)
        min_trade_units = self.config_manager.STRATEGY_MIN_TRADE_UNITS_ASSET
        max_position_units_abs = self.config_manager.STRATEGY_MAX_POSITION_UNITS_ASSET
        if abs_proposed_units < min_trade_units and abs_proposed_units > FLOAT_COMPARISON_TOLERANCE:
            return False, 0.0, f"交易单位 {abs_proposed_units:.8f} < 最小限制 {min_trade_units:.8f}"

        current_pos = self.portfolio.current_position_units
        potential_new_pos_abs = abs(current_pos + proposed_units)
        adjusted_units = proposed_units

        if potential_new_pos_abs > max_position_units_abs + FLOAT_COMPARISON_TOLERANCE:
            # logger.warning(f"提议交易 {proposed_units:.6f} 可能超仓 (当前 {current_pos:.6f}, 限制 {max_position_units_abs:.6f})")
            if proposed_units > 0:
                allowed_units_change = max_position_units_abs - current_pos
                adjusted_units = min(proposed_units, allowed_units_change);
                adjusted_units = max(0.0, adjusted_units)
            else:
                allowed_units_change = -max_position_units_abs - current_pos
                adjusted_units = max(proposed_units, allowed_units_change);
                adjusted_units = min(0.0, adjusted_units)

            if abs(adjusted_units) < min_trade_units and abs(adjusted_units) > FLOAT_COMPARISON_TOLERANCE:
                msg = f"调整后单位 {adjusted_units:.8f} < 最小 {min_trade_units:.8f} (原 {proposed_units:.8f}). 无法执行."
                logger.warning(msg)
                return False, 0.0, msg
            if not np.isclose(proposed_units, adjusted_units):
                logger.info(f"交易单位从 {proposed_units:.6f} 调整为 {adjusted_units:.6f} 以满足持仓限制.")
            return True, adjusted_units, f"已调整 ({adjusted_units:.6f})"
        return True, proposed_units, "通过验证"

    def check_trade_cooldown(self) -> bool:
        if self.portfolio.last_trade_timestamp is None: return True
        cooldown_seconds = self.config_manager.STRATEGY_TRADE_COOLDOWN_SECONDS
        time_since_last_trade = (
                    pd.Timestamp.now(tz=timezone.utc) - self.portfolio.last_trade_timestamp).total_seconds()
        if time_since_last_trade < cooldown_seconds:
            logger.debug(f"交易冷却中: {cooldown_seconds - time_since_last_trade:.1f}s 剩余")
            return False
        return True

    def check_flip_interval(self) -> bool:
        if self.portfolio.last_flip_timestamp is None: return True
        min_flip_interval = self.config_manager.STRATEGY_MIN_FLIP_INTERVAL_SECONDS
        time_since_last_flip = (pd.Timestamp.now(tz=timezone.utc) - self.portfolio.last_flip_timestamp).total_seconds()
        if time_since_last_flip < min_flip_interval:
            logger.debug(f"反手冷却中: {min_flip_interval - time_since_last_flip:.1f}s 剩余")
            return False
        return True

    def is_trading_allowed(self, proposed_units: float = 0.0) -> Tuple[bool, str]:
        if self.is_trading_halted_daily_loss: return False, "日亏损暂停"
        if self.is_trading_halted_consecutive_loss: return False, "连亏暂停"
        if self.is_trading_halted_drawdown: return False, "回撤暂停"
        if not self.check_daily_loss_limit(): return False, "日亏损限制"
        if not self.check_consecutive_loss_limit(): return False, "连亏限制"
        if not self.check_drawdown_limit(): return False, "回撤限制"

        if abs(proposed_units) > FLOAT_COMPARISON_TOLERANCE:
            if not self.check_trade_cooldown(): return False, "交易冷却"
            is_potential_flip = (abs(self.portfolio.current_position_units) > FLOAT_COMPARISON_TOLERANCE and
                                 proposed_units * self.portfolio.current_position_units < -FLOAT_COMPARISON_TOLERANCE)
            if is_potential_flip and not self.check_flip_interval(): return False, "反手间隔限制"
        return True, "允许交易"

    def reset_daily_flags(self):
        self.is_trading_halted_daily_loss = False
        self.is_trading_halted_drawdown = False
        self.daily_peak_equity = self.portfolio.current_equity
        logger.info(f"风控日内标志已重置。日峰值权益更新为: {self.daily_peak_equity:.2f}")
        self._log_current_limits()

    def get_risk_status(self) -> Dict[str, Any]:
        current_equity = self.portfolio.current_equity;
        drawdown_pct = 0.0
        if self.daily_peak_equity > FLOAT_COMPARISON_TOLERANCE and current_equity < self.daily_peak_equity:
            drawdown = self.daily_peak_equity - current_equity
            if drawdown > 0: drawdown_pct = (drawdown / self.daily_peak_equity) * 100

        session_loss_pct = 0.0
        if self.portfolio.initial_cash > FLOAT_COMPARISON_TOLERANCE and self.portfolio.realized_pnl_session < 0:
            session_loss_pct = abs(self.portfolio.realized_pnl_session) / self.portfolio.initial_cash * 100
        max_dd_limit_val = self.config_manager.get('STRATEGY_MAX_DRAWDOWN_PCT')
        max_dd_display = f"{max_dd_limit_val * 100:.1f}%" if max_dd_limit_val is not None else "N/A"
        return {
            'trading_halted_flags': {
                'daily_loss': self.is_trading_halted_daily_loss,
                'consecutive_loss': self.is_trading_halted_consecutive_loss,
                'drawdown': self.is_trading_halted_drawdown,
            },
            'consecutive_losses_count': self.portfolio.consecutive_losses,
            'max_consecutive_losses_limit': self.config_manager.STRATEGY_MAX_CONSECUTIVE_LOSSES,
            'session_loss_pct': session_loss_pct,
            'max_daily_loss_pct_limit': self.config_manager.STRATEGY_MAX_DAILY_LOSS_PCT * 100,
            'current_drawdown_from_peak_pct': drawdown_pct,
            'max_drawdown_pct_limit': max_dd_display,
            'daily_peak_equity': self.daily_peak_equity,
        }


# --- LOGICAL FILE: strategy_service.py (Main Class) ---
class StrategyService:
    def __init__(self, trading_mode: TradingMode = TradingMode.PAPER):
        self.trading_mode: TradingMode = trading_mode
        self.state: StrategyState = StrategyState.STOPPED
        try:
            self.config_manager: StrategyConfigManager = StrategyConfigManager()
        except ValueError as e:
            logger.critical(f"策略服务初始化失败：配置管理器错误 - {e}", exc_info=True)
            self.state = StrategyState.ERROR
            raise RuntimeError(f"无法初始化策略服务，配置错误: {e}") from e

        self.signal_generator: SignalGenerator = SignalGenerator()
        self.portfolio: PortfolioState = PortfolioState(
            asset_pair=str(self.config_manager.STRATEGY_ASSET_PAIR),
            config_manager=self.config_manager
        )
        self.risk_manager: RiskManager = RiskManager(self.portfolio, self.config_manager)
        self.prediction_cache: Dict[str, deque[PredictionResult]] = {}
        self.recent_signals: deque[TradeSignal] = deque(maxlen=500)
        self.execution_queue: deque[TradeSignal] = deque()
        self._lock: threading.RLock = threading.RLock()
        self._shutdown_event: threading.Event = threading.Event()
        self._strategy_thread: Optional[threading.Thread] = None
        self.last_heartbeat: pd.Timestamp = pd.Timestamp.now(tz=timezone.utc)
        self.performance_metrics: Dict[str, int] = {
            'signals_generated': 0, 'trades_executed': 0, 'risk_blocks': 0,
            'errors_signal_generation': 0, 'errors_trade_execution': 0,
        }
        logger.info(
            f"策略服务已初始化 - 模式: {self.trading_mode.value}, "
            f"资产: {self.config_manager.STRATEGY_ASSET_PAIR}"
        )

    def add_prediction(self, prediction: PredictionResult):
        with self._lock:
            model_name = prediction.model_name
            if model_name not in self.prediction_cache:
                pred_cache_maxlen = self.config_manager.get("STRATEGY_PREDICTION_CACHE_MAXLEN", 100)
                self.prediction_cache[model_name] = deque(maxlen=pred_cache_maxlen)
            self.prediction_cache[model_name].append(prediction)

            # 修改这里的日志格式化
            conf_display = f"{prediction.confidence:.2f}" if prediction.confidence is not None else "N/A"
            logret_display = f"{prediction.pred_logret:.4f}" if prediction.pred_logret is not None else "N/A"
            logger.debug(
                f"已添加 {model_name} 预测 (Conf: {conf_display}, LogRet: {logret_display}). "
                f"缓存大小: {len(self.prediction_cache[model_name])}"
            )

    def _cleanup_expired_predictions(self, model_to_clean: Optional[str] = None):
        # now = pd.Timestamp.now(tz=timezone.utc) # is_expired 内部会获取
        models_to_iterate = [model_to_clean] if model_to_clean else list(self.prediction_cache.keys())
        for model_name in models_to_iterate:
            if model_name not in self.prediction_cache: continue
            current_preds = self.prediction_cache[model_name];
            cleaned_count = 0
            while current_preds and current_preds[0].is_expired:
                current_preds.popleft();
                cleaned_count += 1
            if cleaned_count > 0:
                logger.debug(f"从 '{model_name}' 清理了 {cleaned_count} 过期预测. 剩余: {len(current_preds)}")

    def _get_latest_valid_predictions(self) -> Dict[str, PredictionResult]:
        valid_predictions: Dict[str, PredictionResult] = {}
        with self._lock:
            for model_name, predictions_deque in self.prediction_cache.items():
                for i in range(len(predictions_deque) - 1, -1, -1):
                    prediction = predictions_deque[i]
                    if not prediction.is_expired:
                        valid_predictions[model_name] = prediction;
                        break
        return valid_predictions

    def _generate_combined_signal(self, current_market_price: float) -> Optional[TradeSignal]:
        try:
            latest_valid_preds = self._get_latest_valid_predictions()
            min_models = self.config_manager.get("STRATEGY_MIN_MODELS_FOR_SIGNAL", 2)
            if len(latest_valid_preds) < min_models:
                logger.debug(f"有效预测模型 {len(latest_valid_preds)} < 所需 {min_models}，不生成信号")
                return None

            total_weight, weighted_logret_sum = 0.0, 0.0
            source_models, raw_details = [], {}
            model_weights_cfg = {
                "A2": self.config_manager.STRATEGY_WEIGHT_A2,
                "A201": self.config_manager.STRATEGY_WEIGHT_A201,
            }
            default_w = self.config_manager.get("STRATEGY_DEFAULT_WEIGHT_OTHER_MODELS", 0.0)
            use_conf_w = self.config_manager.get("STRATEGY_USE_CONFIDENCE_WEIGHTING", False)
            min_conf_w = self.config_manager.get("STRATEGY_MIN_CONFIDENCE_FOR_WEIGHTING", 0.3)

            for model_name, prediction in latest_valid_preds.items():
                base_w = model_weights_cfg.get(model_name, default_w)
                final_w = base_w
                if use_conf_w and prediction.confidence is not None:
                    final_w *= prediction.confidence if prediction.confidence >= min_conf_w else 0.0
                if final_w > FLOAT_COMPARISON_TOLERANCE and prediction.pred_logret is not None:
                    weighted_logret_sum += final_w * prediction.pred_logret
                    total_weight += final_w;
                    source_models.append(model_name)
                    raw_details[model_name] = {'logret': prediction.pred_logret, 'conf': prediction.confidence,
                                               'weight': final_w}

            if total_weight < FLOAT_COMPARISON_TOLERANCE:
                logger.debug("总权重过低，不生成信号。")
                return None

            avg_logret = weighted_logret_sum / total_weight
            str_raw = abs(avg_logret)
            str_div = self.config_manager.get("STRATEGY_LOGRET_TO_STRENGTH_DIVISOR", 0.02)
            str_calib = np.clip(str_raw / str_div, 0.0, 1.0)

            if str_calib < self.config_manager.STRATEGY_SIGNAL_THRESHOLD_COMBINED:
                logger.debug(
                    f"信号强度 {str_calib:.4f} < 阈值 {self.config_manager.STRATEGY_SIGNAL_THRESHOLD_COMBINED:.4f}")
                return None

            sig_type = "BUY" if avg_logret > 0 else "SELL"
            units_abs: float
            if self.config_manager.get('STRATEGY_USE_FIXED_TRADE_SIZE_USDT', True):
                val_usdt = self.config_manager.STRATEGY_FIXED_TRADE_SIZE_USDT
                if current_market_price <= FLOAT_COMPARISON_TOLERANCE:
                    logger.error(f"市价 {current_market_price} 过低，无法计算USDT固定大小单位。");
                    return None
                units_abs = val_usdt / current_market_price
            else:
                min_u = self.config_manager.STRATEGY_MIN_TRADE_UNITS_ASSET
                dyn_f = self.config_manager.get("STRATEGY_DYNAMIC_UNITS_MAX_FACTOR", 5.0)
                units_abs = min_u + (min_u * (dyn_f - 1) * str_calib)

            cur_pos, max_pos_abs = self.portfolio.current_position_units, self.config_manager.STRATEGY_MAX_POSITION_UNITS_ASSET
            if (sig_type == "BUY" and cur_pos >= max_pos_abs - FLOAT_COMPARISON_TOLERANCE) or \
                    (sig_type == "SELL" and cur_pos <= -max_pos_abs + FLOAT_COMPARISON_TOLERANCE):
                logger.info(f"已达 {sig_type} 方向最大仓位，不生成新开仓信号。");
                return None

            prop_units: float
            is_flip = (sig_type == "BUY" and cur_pos < -FLOAT_COMPARISON_TOLERANCE) or \
                      (sig_type == "SELL" and cur_pos > FLOAT_COMPARISON_TOLERANCE)
            prop_units = (abs(cur_pos) + units_abs) if is_flip else units_abs
            if sig_type == "SELL": prop_units = -prop_units

            is_valid_size, final_units, reason_size = self.risk_manager.validate_trade_size(prop_units)
            if not is_valid_size: logger.warning(
                f"交易单位验证失败: {reason_size}. 原提议: {prop_units:.6f}"); return None
            if abs(final_units) < FLOAT_COMPARISON_TOLERANCE: logger.info(
                f"交易单位调整后为0 ({reason_size})"); return None

            sl_price, tp_price = None, None
            sl_pct, tp_rr = self.config_manager.get('STRATEGY_STOP_LOSS_PERCENTAGE'), self.config_manager.get(
                'STRATEGY_TAKE_PROFIT_RR_RATIO')
            if sl_pct is not None:
                sl_amt = current_market_price * sl_pct
                if sig_type == "BUY":
                    sl_price = current_market_price - sl_amt
                else:
                    sl_price = current_market_price + sl_amt
                if tp_rr is not None and sl_price is not None:  # Ensure sl_price is calculated
                    if sig_type == "BUY":
                        tp_price = current_market_price + (current_market_price - sl_price) * tp_rr
                    else:
                        tp_price = current_market_price - (sl_price - current_market_price) * tp_rr

            urg = "NORMAL"
            urg_h, urg_c = self.config_manager.get("STRATEGY_URGENCY_THRESHOLD_HIGH", 0.015), self.config_manager.get(
                "STRATEGY_URGENCY_THRESHOLD_CRITICAL", 0.025)
            if str_raw >= urg_c:
                urg = "CRITICAL"
            elif str_raw >= urg_h:
                urg = "HIGH"

            trade_signal = TradeSignal(
                pd.Timestamp.now(tz=timezone.utc), source_models, self.portfolio.asset_pair, sig_type,
                str_calib, final_units, current_market_price, sl_price, tp_price,
                f"Combined: ALR={avg_logret:.5f}, StrCal={str_calib:.3f}", raw_details, urg
            )
            self.performance_metrics['signals_generated'] += 1
            logger.info(f"生成交易信号: {sig_type} {final_units:.4f} @ {current_market_price:.2f}, Str={str_calib:.3f}")
            return trade_signal
        except Exception as e:
            logger.error(f"生成综合信号时出错: {e}", exc_info=True)
            self.performance_metrics['errors_signal_generation'] += 1
            return None

    def process_market_data(self, market_price: float, timestamp: Optional[pd.Timestamp] = None):
        if self.state == StrategyState.STOPPED or self.state == StrategyState.ERROR:
            return
        ts_proc = timestamp if timestamp else pd.Timestamp.now(tz=timezone.utc)
        with self._lock:
            self.portfolio.update_unrealized_pnl(market_price)
            if self.state == StrategyState.ACTIVE:
                signal = self._generate_combined_signal(market_price)
                if signal:
                    self.recent_signals.append(signal)
                    is_allowed, reason = self.risk_manager.is_trading_allowed(signal.execution_units)
                    if is_allowed:
                        self.execution_queue.append(signal)
                        logger.info(
                            f"信号入队 ({len(self.execution_queue)}): {signal.signal_type} {abs(signal.execution_units):.4f}")
                    else:
                        self.performance_metrics['risk_blocks'] += 1
                        logger.warning(f"信号被风控阻止 ({reason}): {signal.signal_type} Str={signal.strength:.3f}")
            self.last_heartbeat = ts_proc

    def execute_pending_signals(self, current_market_price: float) -> List[Dict[str, Any]]:
        executed_trades: List[Dict[str, Any]] = []
        if self.state != StrategyState.ACTIVE:
            if self.execution_queue: logger.warning(
                f"策略状态 {self.state.value}, 不执行队列中 {len(self.execution_queue)} 信号。")
            return executed_trades
        with self._lock:
            if any(val for key, val in self.risk_manager.get_risk_status()['trading_halted_flags'].items()):
                if self.execution_queue: logger.critical(
                    f"全局交易暂停，清空执行队列 {len(self.execution_queue)} 信号。"); self.execution_queue.clear()
                return executed_trades

            proc_run = 0;
            max_proc = self.config_manager.get("STRATEGY_MAX_SIGNALS_PER_EXECUTION", 5)
            while self.execution_queue and proc_run < max_proc:
                signal = self.execution_queue.popleft();
                proc_run += 1
                try:
                    is_allowed, reason = self.risk_manager.is_trading_allowed(signal.execution_units)
                    if not is_allowed:
                        logger.warning(
                            f"执行信号时风控阻止 ({reason}): {signal.signal_type} {signal.execution_units:.4f}")
                        self.performance_metrics['risk_blocks'] += 1;
                        continue

                    exec_price = current_market_price  # Simplified execution price
                    trade_res = self._execute_live_trade(signal,
                                                         exec_price) if self.trading_mode == TradingMode.LIVE else self._execute_paper_trade(
                        signal, exec_price)
                    if trade_res.get('success', False):
                        executed_trades.append(trade_res);
                        self.performance_metrics['trades_executed'] += 1
                    else:
                        logger.error(f"交易执行失败: {trade_res.get('error', '未知')}. 信号: {signal.signal_type}")
                        self.performance_metrics['errors_trade_execution'] += 1
                except Exception as e:
                    logger.error(f"执行交易信号严重错误: {e}. 信号: {signal.signal_type}", exc_info=True)
                    self.performance_metrics['errors_trade_execution'] += 1
        if executed_trades: logger.info(f"本轮执行 {len(executed_trades)} 笔交易。")
        return executed_trades

    def _execute_paper_trade(self, signal: TradeSignal, execution_price: float) -> Dict[str, Any]:
        try:
            self.portfolio.update_position(signal.execution_units, execution_price)
            return {
                'success': True, 'signal_id': f"{signal.timestamp.strftime('%Y%m%d%H%M%S%f')}-{signal.signal_type[:1]}",
                'asset_pair': signal.asset_pair, 'trade_type': signal.signal_type, 'units': signal.execution_units,
                'price': execution_price, 'timestamp': pd.Timestamp.now(tz=timezone.utc).isoformat(),
                'reason': signal.reason[:50],
            }
        except Exception as e:
            logger.error(f"模拟交易执行出错: {e}", exc_info=True)
            return {'success': False, 'error': str(e), 'signal_id': id(signal)}

    def _execute_live_trade(self, signal: TradeSignal, estimated_price: float) -> Dict[str, Any]:
        logger.warning("实盘交易功能 (_execute_live_trade) 未实现，使用模拟逻辑。")
        actual_price, actual_units = estimated_price, signal.execution_units
        try:
            self.portfolio.update_position(actual_units, actual_price)
            return {
                'success': True, 'order_id': f"live_{pd.Timestamp.now(tz=timezone.utc).timestamp()}",
                'asset_pair': signal.asset_pair, 'trade_type': signal.signal_type,
                'requested_units': signal.execution_units,
                'executed_units': actual_units, 'avg_price': actual_price, 'status': "FILLED",
                'timestamp': pd.Timestamp.now(tz=timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"实盘交易（模拟）更新持仓出错: {e}", exc_info=True)
            return {'success': False, 'error': f"Live portfolio update error: {str(e)}"}

    def start(self):
        with self._lock:
            if self.state not in [StrategyState.STOPPED, StrategyState.ERROR]:
                logger.warning(f"策略已运行或暂停 (状态: {self.state.value})，无法重复启动。");
                return
            if self.state == StrategyState.ERROR: logger.warning("策略处错误状态，尝试重置启动。")
            try:
                if not hasattr(self, 'config_manager') or self.config_manager is None:
                    self.config_manager = StrategyConfigManager()  # Should not happen
                    self.config_manager.load_initial_config()
                self.config_manager.start_file_watcher()
                self.portfolio = PortfolioState(str(self.config_manager.STRATEGY_ASSET_PAIR), self.config_manager)
                self.risk_manager = RiskManager(self.portfolio, self.config_manager)
                self.prediction_cache.clear();
                self.recent_signals.clear();
                self.execution_queue.clear()
                self.performance_metrics = {k: 0 for k in self.performance_metrics}
            except (ValueError, RuntimeError) as e:
                logger.critical(f"策略启动失败：组件初始化错误 - {e}", exc_info=True)
                self.state = StrategyState.ERROR
                if hasattr(self, 'config_manager') and self.config_manager: self.config_manager.shutdown()
                return
            self.state = StrategyState.ACTIVE;
            self._shutdown_event.clear()
            if self._strategy_thread is None or not self._strategy_thread.is_alive():
                self._strategy_thread = threading.Thread(target=self._strategy_loop, daemon=True,
                                                         name="StrategyMainLoop")
                self._strategy_thread.start();
                logger.info("策略主循环线程已启动。")
            else:
                logger.info("策略主循环线程已运行，状态置为ACTIVE。")
            logger.info(f"策略服务已启动。模式: {self.trading_mode.value}")

    def stop(self):
        with self._lock:
            if self.state == StrategyState.STOPPED: logger.info("策略服务已经停止。"); return
            logger.info(f"正在停止策略服务 (当前状态: {self.state.value})...");
            self.state = StrategyState.STOPPED
            self._shutdown_event.set()
            if self._strategy_thread and self._strategy_thread.is_alive():
                logger.info("等待策略主循环线程退出...");
                self._strategy_thread.join(timeout=10.0)
                if self._strategy_thread.is_alive(): logger.warning("策略主循环线程超时未停止。")
            if hasattr(self, 'config_manager') and self.config_manager: self.config_manager.shutdown()
            logger.info("策略服务已停止。")

    def pause(self):
        with self._lock:
            if self.state == StrategyState.ACTIVE:
                self.state = StrategyState.PAUSED; logger.info("策略服务已暂停。")
            elif self.state == StrategyState.PAUSED:
                logger.info("策略已暂停。")
            else:
                logger.warning(f"策略状态 {self.state.value}，无法暂停。")

    def resume(self):
        with self._lock:
            if self.state == StrategyState.PAUSED:
                self.state = StrategyState.ACTIVE;
                self.last_heartbeat = pd.Timestamp.now(tz=timezone.utc)
                logger.info("策略服务已从暂停恢复。")
            elif self.state == StrategyState.ACTIVE:
                logger.info("策略已活动。")
            else:
                logger.warning(f"策略状态 {self.state.value}，无法恢复。")

    def _strategy_loop(self):
        logger.info(f"策略主循环 ({threading.current_thread().name}) 已启动。")
        loop_interval = float(
            self.config_manager.get("STRATEGY_MAIN_LOOP_INTERVAL_SECONDS", STRATEGY_LOOP_TIMEOUT_SECONDS))
        while not self._shutdown_event.is_set():
            try:
                current_time = pd.Timestamp.now(tz=timezone.utc)
                with self._lock:
                    self._cleanup_expired_predictions()
                    if self.state in [StrategyState.ACTIVE, StrategyState.PAUSED]:
                        hb_timeout = float(
                            self.config_manager.get("STRATEGY_HEARTBEAT_TIMEOUT_SECONDS", HEARTBEAT_TIMEOUT_SECONDS))
                        if (current_time - self.last_heartbeat).total_seconds() > hb_timeout:
                            logger.warning(
                                f"长时间 ({(current_time - self.last_heartbeat).total_seconds():.0f}s) 未收市价心跳。")
            except Exception as e:
                logger.error(f"策略主循环严重错误: {e}", exc_info=True)
                time.sleep(max(1.0, loop_interval / 2))
            if self._shutdown_event.wait(timeout=loop_interval): break
        logger.info(f"策略主循环 ({threading.current_thread().name}) 已退出。")

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            try:
                cfg = self.config_manager
                asset_p, sig_t, max_p = cfg.STRATEGY_ASSET_PAIR, cfg.STRATEGY_SIGNAL_THRESHOLD_COMBINED, cfg.STRATEGY_MAX_POSITION_UNITS_ASSET
                dyn_mtime_val = cfg._last_modified_time
                dyn_mtime_str = datetime.fromtimestamp(dyn_mtime_val,
                                                       tz=timezone.utc).isoformat() if dyn_mtime_val > 0 else "N/A"
                cfg_ref_int = cfg.get('STRATEGY_CONFIG_REFRESH_INTERVAL_SECONDS', 'N/A (default)')
            except AttributeError as e:
                logger.warning(f"获取状态时关键配置缺失: {e}")
                asset_p, sig_t, max_p, dyn_mtime_str, cfg_ref_int = "Err", "Err", "Err", "Err", "Err"

            main_alive = self._strategy_thread.is_alive() if self._strategy_thread else False
            cfg_w_alive = self.config_manager._file_watcher_thread.is_alive() if self.config_manager._file_watcher_thread else False
            return {
                'service_status': self.state.value, 'trading_mode': self.trading_mode.value,
                'timestamp_utc': pd.Timestamp.now(tz=timezone.utc).isoformat(),
                'last_market_data_heartbeat_utc': self.last_heartbeat.isoformat(),
                'time_since_last_heartbeat_seconds': round(
                    (pd.Timestamp.now(tz=timezone.utc) - self.last_heartbeat).total_seconds(), 2),
                'portfolio_summary': self.portfolio.get_performance_summary(),
                'risk_status': self.risk_manager.get_risk_status(),
                'performance_metrics': self.performance_metrics.copy(),
                'prediction_cache_info': {'models': len(self.prediction_cache),
                                          'counts': {k: len(v) for k, v in self.prediction_cache.items()}},
                'recent_signals_count': len(self.recent_signals), 'execution_queue_size': len(self.execution_queue),
                'config_summary': {'asset': asset_p, 'sig_thresh': sig_t, 'max_pos': max_p, 'dyn_mtime': dyn_mtime_str,
                                   'refresh_interval': cfg_ref_int, },
                'threads_alive': {'main_loop': main_alive, 'config_watcher': cfg_w_alive, }}

    def force_reset_risk_controls(self, reset_consecutive_losses: bool = False, reset_session_pnl: bool = False):
        with self._lock:
            logger.warning("!!! 强制重置风控标志和状态 !!!")
            self.risk_manager.reset_daily_flags()
            if reset_consecutive_losses: self.portfolio.consecutive_losses = 0; self.portfolio.consecutive_wins = 0; logger.warning(
                "连亏/盈计数已重置。")
            if reset_session_pnl: self.portfolio.realized_pnl_session = 0.0; logger.warning("会话已实现盈亏已重置。")
            logger.info("风控标志和状态已强制重置。")

    def __del__(self):
        logger.debug(f"StrategyService ({id(self)}) 正在垃圾回收。")
        if self.state != StrategyState.STOPPED:
            logger.warning(f"StrategyService ({id(self)}) 未正确停止就被销毁。State: {self.state.value}")


def create_strategy_service(trading_mode: str = "PAPER") -> StrategyService:
    try:
        mode = TradingMode(trading_mode.upper())
    except ValueError:
        valid_modes = [m.value for m in TradingMode]
        logger.error(f"无效交易模式: '{trading_mode}'. 有效: {valid_modes}")
        raise ValueError(f"无效交易模式: '{trading_mode}'. 有效: {valid_modes}")
    return StrategyService(trading_mode=mode)


if __name__ == "__main__":
    log_level_str = os.environ.get("STRATEGY_LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler()])

    logger.info("策略服务主程序启动...")
    dynamic_config_path = Path(config.STRATEGY_DYNAMIC_CONFIG_FILE_PATH)
    # 使用符合规则的示例动态配置
    dynamic_config_content_example = {
        "STRATEGY_WEIGHT_A2": 0.6,  # 确保和 A201 的和为 1 (A201 默认为 0.4)
        "STRATEGY_WEIGHT_A201": 0.4,  # 或者也在这里定义
        "STRATEGY_SIGNAL_THRESHOLD_COMBINED": 0.007,
        "STRATEGY_MAX_DAILY_LOSS_PCT": 0.03,  # 在范围内
        "STRATEGY_FIXED_TRADE_SIZE_USDT": 1500.0,
    }
    try:
        with open(dynamic_config_path, 'w', encoding='utf-8') as f_dyn:
            json.dump(dynamic_config_content_example, f_dyn, indent=2)
        logger.info(f"创建/更新了动态配置文件: {dynamic_config_path.resolve()}")
    except IOError as e:
        logger.error(f"无法写入动态配置文件 {dynamic_config_path}: {e}")

    strategy_service_instance: Optional[StrategyService] = None
    try:
        strategy_service_instance = create_strategy_service(trading_mode="PAPER")
        strategy_service_instance.start()
        logger.info("策略服务已请求启动。")
        time.sleep((config.STRATEGY_CONFIG_REFRESH_INTERVAL_SECONDS / 5) if hasattr(config,
                                                                                    'STRATEGY_CONFIG_REFRESH_INTERVAL_SECONDS') else 3)

        logger.info("模拟添加预测数据...")
        pred_time = pd.Timestamp.now(tz=timezone.utc) - pd.Timedelta(seconds=10)
        common_args = {"ts_pred": pred_time, "horizon_sec": 300, "base_price": 50000.0}
        strategy_service_instance.add_prediction(
            PredictionResult(model_name="A2", pred_logret=0.01, confidence=0.85, **common_args))
        strategy_service_instance.add_prediction(
            PredictionResult(model_name="A201", pred_logret=0.009, confidence=0.70, **common_args))

        current_market_price = 50100.0
        logger.info(f"模拟市场价格更新: {current_market_price}")
        strategy_service_instance.process_market_data(current_market_price)
        time.sleep(0.5)
        logger.info("模拟执行挂单...")
        executed_trades = strategy_service_instance.execute_pending_signals(current_market_price)
        if executed_trades:
            logger.info(f"执行的交易: {json.dumps(executed_trades, indent=2)}")
        else:
            logger.info("本轮无交易执行。")

        time.sleep(0.5);
        status = strategy_service_instance.get_status()
        logger.info(f"当前策略状态: \n{json.dumps(status, indent=2, default=str)}")

        logger.info("模拟动态修改配置文件 (降低阈值)...")
        dynamic_config_content_example["STRATEGY_SIGNAL_THRESHOLD_COMBINED"] = 0.001
        dynamic_config_content_example["STRATEGY_TRADE_COOLDOWN_SECONDS"] = 5
        try:
            with open(dynamic_config_path, 'w', encoding='utf-8') as f_dyn:
                json.dump(dynamic_config_content_example, f_dyn, indent=2)
            refresh_interval = strategy_service_instance.config_manager.get('STRATEGY_CONFIG_REFRESH_INTERVAL_SECONDS',
                                                                            60)
            logger.info(f"等待配置刷新 (约 {refresh_interval + 2} 秒)...")
            time.sleep(refresh_interval + 2)
            status_after_change = strategy_service_instance.get_status()
            new_thresh = status_after_change['config_summary']['sig_thresh']
            logger.info(f"配置更改后信号阈值: {new_thresh} (期望接近 0.001)")
            strategy_service_instance.process_market_data(current_market_price + 10)  # 触发新信号
            time.sleep(0.2);
            strategy_service_instance.execute_pending_signals(current_market_price + 10)
        except IOError as e:
            logger.error(f"无法写入动态配置文件测试: {e}")
        logger.info("模拟运行10秒...")
        time.sleep(10)
    except RuntimeError as e:
        logger.critical(f"无法启动或运行策略服务: {e}", exc_info=True)
    except KeyboardInterrupt:
        logger.info("用户中断 (Ctrl+C)...")
    except Exception as e:
        logger.critical(f"主程序未捕获错误: {e}", exc_info=True)
    finally:
        if strategy_service_instance:
            logger.info("正在停止策略服务...");
            strategy_service_instance.stop()
            logger.info("策略服务已请求停止。等待线程结束...");
            time.sleep(1)
        logger.info("策略服务主程序结束。")

# --- END OF FILE strategy_services.py ---