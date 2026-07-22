# grid-backtest — A股网格交易回测系统

## 快速开始

```bash
cd grid-backtest

# 单 ETF 回测（使用默认参数）
python -m src.main --symbol 510300 --market SH

# 使用预设模板
python -m src.main --symbol 510300 --preset moderate

# 自定义网格参数
python -m src.main --symbol 002036 --market SZ --step 0.02 --grids 20 --pos 0.05

# 参数优化
python -m src.main --symbol 510300 --optimize --space standard

# 批量 ETF 池回测
python -m src.main --pool etf

# 批量 ETF 池优化
python -m src.main --pool etf --optimize

# 列出现有预设策略
python -m src.main --list-presets
```

## 运行测试

```bash
python -m pytest tests/ -v    # 17 个测试
```

## 依赖

```bash
pip install pandas numpy matplotlib efinance akshare openpyxl pytest
```

## 产出位置

所有图表和报告生成在 `output/` 目录：
- `grid_equity_<symbol>.png` — 权益曲线 + 交易标记
- `grid_price_<symbol>.png` — 价格走势 + 网格线叠加
- `grid_heatmap_<symbol>.png` — 参数优化热力图
- `grid_multi_comparison.png` — 多标的对比图
- `grid_report_<symbol>.html` — 完整的 HTML 报告
- `grid_results_summary.xlsx` — Excel 汇总 + 交易明细
- `optimization_<symbol>.md` — 优化报告
