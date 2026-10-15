# BTC/USDT short-horizon prediction & execution system

A research project that predicts BTC/USDT short-horizon price moves from market and on-chain features and trades on OKX.

> **Status:** research / learning project. It is not investment advice, and nothing here claims profitability. Use the demo trading environment.

## Architecture

![Architecture](docs/architecture.png)

```
PostgreSQL (market + on-chain features)
        │
        ▼
data_services.py ──► feature pipelines (a1.py: 30s, a101.py: 3min log-return)
        │
        ▼
models: a2.py (Hybrid Neural ODE + wavelets) · a201.py (TimesNet-style regressor)
        │            learning_services.py: online learning, error tracking
        ▼
strategy_services.py ──► signal blending, risk manager (daily loss cap, cooldown)
        │
        ▼
exchange_handler.py ──► OKX REST + private WebSocket, multi-account router
        │
trading_orchestrator.py  (MasterController: wires it all together)
```

| Module | What it does |
|---|---|
| `a1.py` | 30-second feature processor |
| `core_utils.py` | Rolling feature/raw caches, EMA, signal generator |
| `a101.py` | 3-minute resampling of market and on-chain data, log-return targets |
| `a2.py` | Hybrid Neural ODE price model with wavelet decomposition (PyTorch, `dopri5`) |
| `a201.py` | Price-change regressor with bucketed batching and attention pooling |
| `data_services.py` | Data pipeline manager and model prediction service |
| `learning_services.py` | Online learning manager and background error sweeper |
| `strategy_services.py` | Strategy state, signal combination, portfolio state, risk management |
| `exchange_handler.py` | Async OKX REST client (httpx), private WebSocket, account routing |
| `trading_orchestrator.py` | Top-level controller |

## Setup

Requirements: Python 3.10+, PostgreSQL, PyTorch.

Credentials come from environment variables. Never hardcode them.

```bash
export DATABASE_URL="postgresql+psycopg2://USER:PASSWORD@localhost:5432/crypto_data"
# or DB_USER / DB_PASS separately
export OKX_API_KEY=...        # put these in a local .env (ignored by git)
export OKX_API_SECRET=...
export OKX_API_PASSPHRASE=...
export TRADING_ENV=demo       # OKX demo trading
```

Strategy parameters live in `strategy_params_live.json` and `config.py`.

## Notes

- `*.pth` / `*.joblib` files are trained weights and scalers for the model checkpoints referenced in `config.py`.
- The code comments are partly in Chinese.
