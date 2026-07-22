# Quant System — Multi-Role Quantitative Trading Framework

> Inspired by the 15-role institutional quant pipeline at Goldman Sachs, Renaissance, Citadel, Two Sigma, AQR, Man Group, and others.

## Overview

Quant System implements a complete quantitative trading research pipeline organized around 6 institutional roles:

| Role | Module | What It Does |
|------|--------|--------------|
| 1. Strategy Architect | `strategy/` | Design strategies as YAML specs with factor exposures, constraints, and regime gates |
| 2. Backtest Engine | `backtest/` | Multi-stock portfolio backtesting with walk-forward validation, Monte Carlo, and significance testing |
| 3. Risk Manager | `risk/` | Four VaR methods, stress testing, correlation monitoring, exposure limits |
| 4. Alpha Researcher | `alpha/` | Feature engineering, IC/quantile/Fama-MacBeth testing, signal decay analysis |
| 5. Factor Builder | `factor/` | Momentum, value, quality, size, volatility factors with correlation and attribution |
| 6. Portfolio Optimizer | `portfolio/` | MVO with shrinkage, Black-Litterman, Risk Parity, Hierarchical Risk Parity |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Validate a built-in strategy
python -m quant_system.main architect --template multi_factor

# Run a backtest (dry run with top 20 HS300 stocks)
python -m quant_system.main backtest --strategy momentum

# Generate weekly signals (dry run)
python -m quant_system.main live --dry-run

# Full pipeline with PushPlus push
python -m quant_system.main live --push
```

## Architecture

```
策略架构师 → Alpha研究员 → 因子模型 → 回测引擎 → 风控经理 → 投资组合优化 → Live推送
     ↓            ↓           ↓          ↓          ↓            ↓
  YAML Spec    IC/Decay   多因子合成   组合模拟   VaR/Stress   MVO/HRP/ERC
```

## Configuration

Edit `config.yaml` to customize:
- Stock universe (`hs300`, `csi500`)
- Trading costs (commission, stamp tax, slippage)
- Risk limits (VaR confidence, position limits, drawdown halt)
- Factor model parameters
- PushPlus token (for WeChat notifications)

## Built-in Strategy Templates

| Template | Description | Factors |
|----------|-------------|---------|
| `momentum` | Cross-sectional momentum | 20d, 60d, 120d momentum |
| `value_quality` | Value + Quality | PE, PB, ROE, gross margin |
| `low_vol` | Low volatility anomaly | 20d, 60d realized vol |
| `multi_factor` | Diversified multi-factor | Momentum + Value + Quality + Vol + Size |

## CLI Reference

```bash
# Strategy design
python -m quant_system.main architect --template momentum
python -m quant_system.main architect --spec my_strategy.yaml

# Backtesting
python -m quant_system.main backtest --strategy multi_factor
python -m quant_system.main backtest --strategy momentum --walk-forward --monte-carlo

# Risk analysis
python -m quant_system.main risk --var monte_carlo --stress --dashboard

# Alpha research
python -m quant_system.main alpha --factor momentum_60d --ic-test --decay

# Factor model
python -m quant_system.main factor --compose momentum_60d,value_pe,quality_roe --method erc

# Portfolio optimization
python -m quant_system.main optimize --method hrp

# Live signals
python -m quant_system.main live --dry-run
python -m quant_system.main live --push
```

## Running Tests

```bash
cd c:/AI
python -m pytest quant_system/tests/ -v
```

## Dependencies

- **pandas, numpy, scipy** — Core data and computation
- **efinance, akshare** — A-share data sources
- **matplotlib** — Visualization
- **click** — CLI framework
- **pyyaml** — Configuration parsing
- **requests** — PushPlus API

Optional:
- `hmmlearn` — HMM regime detection
- `scikit-learn` — ML-based signal research

## License

Internal research use.
