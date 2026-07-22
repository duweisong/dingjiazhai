# 实施计划 — grid-backtest

## Phase 1: 核心引擎 ✅
- [x] `config.py` — 配置系统
- [x] `data_loader.py` — 数据加载（efinance + akshare）
- [x] `grid_engine.py` — 网格交易引擎
- [x] `grid_strategy.py` — 预设策略模板
- [x] `optimizer.py` — 参数优化器
- [x] `metrics.py` — 绩效指标
- [x] `visualize.py` — 可视化
- [x] `reporter.py` — 报告生成
- [x] `main.py` — CLI 入口
- [x] 单元测试

## Phase 2: 实盘验证（待做）
- [ ] 在真实 A 股数据上回测
- [ ] 找出网格适用标的特征
- [ ] 参数稳定性分析（不同时间段表现一致性）

## Phase 3: 增强（待做）
- [ ] 动态网格调整（随价格变化重新居中）
- [ ] 分批止盈/止损
- [ ] 网格 + 趋势过滤（MA 闸门）
- [ ] Walk-Forward 分析
