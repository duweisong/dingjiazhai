"""
徐翔体系 · 升级版量化系统
========================

基于"借势打势"内核，升级为适应 2015-2025 市场剧变的现代量化框架。

核心模块:
  market_env      - 市场环境 A/B/C 分级
  capital_struct  - 资金结构分析 (北向/量化/游资/机构/散户)
  scanner         - 选股扫描引擎
  risk_manager    - 五维风控
  strategy        - 升级版策略
  backtest_10y    - 十年回测

用法:
  python -m xu_xiang_system.main scan       # 实时扫描
  python -m xu_xiang_system.main backtest   # 十年回测
"""
__version__ = "1.0.0"
