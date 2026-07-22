import json
from datetime import datetime, timezone

PROJECT_ROOT = "c:/AI/backtest"
PROJECT_NAME = "联创电子A股波段策略回测"
PROJECT_DESC = "联创电子(002036.SZ) A股波段策略回测系统。支持双均线、MACD、布林带、RSI、复合共振等多种策略的历史回测，包含Walk-Forward参数优化、自适应仓位管理、HTML可视化报告。基于A股T+1制度和真实交易成本模拟。"

nodes = [
    # Core Engine
    {"id": "file:main.py", "type": "file", "name": "main.py", "filePath": "main.py",
     "summary": "回测系统主入口。解析命令行参数，调度策略回测、参数优化、滚动窗口优化全流程。",
     "tags": ["入口", "CLI"], "complexity": "moderate"},
    {"id": "file:engine.py", "type": "file", "name": "engine.py", "filePath": "engine.py",
     "summary": "核心回测引擎，模拟A股T+1交易制度。包含Trade、BacktestResult、BacktestEngine三个核心类。处理买卖执行、止损止盈、滑点与成本计算。",
     "tags": ["核心", "T+1"], "complexity": "complex"},
    {"id": "file:config.py", "type": "file", "name": "config.py", "filePath": "config.py",
     "summary": "全局配置：股票002036联创电子、时间范围、初始资金10万、仓位比例、交易成本、所有策略参数。",
     "tags": ["配置"], "complexity": "simple"},
    {"id": "file:data_loader.py", "type": "file", "name": "data_loader.py", "filePath": "data_loader.py",
     "summary": "双数据源行情获取(efinance东方财富 + akshare)，自动回退。计算MA/MACD/RSI/布林带/ATR等技术指标，本地缓存。",
     "tags": ["数据", "技术指标"], "complexity": "moderate"},

    # Strategies
    {"id": "file:strategies/base.py", "type": "file", "name": "strategies/base.py", "filePath": "strategies/base.py",
     "summary": "策略抽象基类。Signal、Position、BaseStrategy(ABC)。所有策略必须实现generate_signals()。",
     "tags": ["基类", "ABC"], "complexity": "simple"},
    {"id": "file:strategies/__init__.py", "type": "file", "name": "strategies/__init__.py", "filePath": "strategies/__init__.py",
     "summary": "策略注册中心。STRATEGIES注册表、get_strategy()工厂函数、list_strategies()。",
     "tags": ["注册"], "complexity": "simple"},
    {"id": "file:strategies/ma_cross.py", "type": "file", "name": "strategies/ma_cross.py", "filePath": "strategies/ma_cross.py",
     "summary": "双均线交叉波段策略。短期均线上穿长期均线买入，下穿卖出。",
     "tags": ["均线", "金叉死叉"], "complexity": "simple"},
    {"id": "file:strategies/macd.py", "type": "file", "name": "strategies/macd.py", "filePath": "strategies/macd.py",
     "summary": "MACD金叉死叉波段策略。DIF上穿DEA买入，下穿卖出，结合零轴位置。",
     "tags": ["MACD", "金叉死叉"], "complexity": "simple"},
    {"id": "file:strategies/bollinger.py", "type": "file", "name": "strategies/bollinger.py", "filePath": "strategies/bollinger.py",
     "summary": "布林带均值回归策略。触及下轨买入，触及上轨卖出。",
     "tags": ["布林带", "均值回归"], "complexity": "simple"},
    {"id": "file:strategies/rsi.py", "type": "file", "name": "strategies/rsi.py", "filePath": "strategies/rsi.py",
     "summary": "RSI超买超卖反转策略。RSI<30买入，RSI>70卖出。",
     "tags": ["RSI", "反转"], "complexity": "simple"},
    {"id": "file:strategies/composite.py", "type": "file", "name": "strategies/composite.py", "filePath": "strategies/composite.py",
     "summary": "多指标共振策略。综合均线+MACD+RSI+布林带，多数信号一致时触发交易。",
     "tags": ["复合", "共振"], "complexity": "moderate"},
    {"id": "file:strategies/advanced_swing.py", "type": "file", "name": "strategies/advanced_swing.py", "filePath": "strategies/advanced_swing.py",
     "summary": "高级波段策略集合：AdvancedSwingStrategy、TrendFollowingStrategy、VolumeBreakoutStrategy。",
     "tags": ["高级", "多时间框架"], "complexity": "complex"},
    {"id": "file:strategies/pattern_swing.py", "type": "file", "name": "strategies/pattern_swing.py", "filePath": "strategies/pattern_swing.py",
     "summary": "K线形态增强策略。结合蜡烛图形态识别和技术指标信号增强可信度。",
     "tags": ["K线形态", "信号增强"], "complexity": "moderate"},

    # Analysis
    {"id": "file:filters.py", "type": "file", "name": "filters.py", "filePath": "filters.py",
     "summary": "市场状态识别(MarketRegimeFilter)、量价分析(VolumePriceAnalyzer)、ATR自适应止损(ATRDynamicStops)。",
     "tags": ["过滤器", "量价"], "complexity": "moderate"},
    {"id": "file:patterns.py", "type": "file", "name": "patterns.py", "filePath": "patterns.py",
     "summary": "蜡烛图形态识别。CandlestickPatterns(锤子线/吞没/十字星等)、ConsecutiveBarAnalyzer。",
     "tags": ["K线形态", "蜡烛图"], "complexity": "moderate"},
    {"id": "file:optimizer.py", "type": "file", "name": "optimizer.py", "filePath": "optimizer.py",
     "summary": "Walk-Forward滚动窗口优化器 + AdaptivePositionSizer(Kelly公式自适应仓位管理)。",
     "tags": ["优化", "Kelly"], "complexity": "complex"},
    {"id": "file:metrics.py", "type": "file", "name": "metrics.py", "filePath": "metrics.py",
     "summary": "回测绩效分析：月/年度收益率、夏普比率、最大回撤、胜率、盈亏比。",
     "tags": ["绩效", "指标"], "complexity": "simple"},
    {"id": "file:visualize.py", "type": "file", "name": "visualize.py", "filePath": "visualize.py",
     "summary": "图表绘制和HTML报告生成：权益曲线、交易分析图、多策略对比。",
     "tags": ["可视化", "报告"], "complexity": "moderate"},

    # Support
    {"id": "file:requirements.txt", "type": "file", "name": "requirements.txt", "filePath": "requirements.txt",
     "summary": "Python依赖：akshare, pandas, numpy, matplotlib, scipy, openpyxl。",
     "tags": ["依赖"], "complexity": "simple"},

    # Key classes
    {"id": "class:engine.py:BacktestEngine", "type": "class", "name": "BacktestEngine", "filePath": "engine.py",
     "summary": "回测执行引擎核心。run()逐日遍历历史数据，执行模拟交易，处理T+1延迟和止损止盈。",
     "tags": ["引擎核心"], "complexity": "complex"},
    {"id": "class:strategies/base.py:BaseStrategy", "type": "class", "name": "BaseStrategy", "filePath": "strategies/base.py",
     "summary": "所有策略的抽象基类。定义generate_signals()接口。",
     "tags": ["策略基类"], "complexity": "simple"},
    {"id": "class:optimizer.py:WalkForwardOptimizer", "type": "class", "name": "WalkForwardOptimizer", "filePath": "optimizer.py",
     "summary": "滚动窗口参数优化器。训练/测试期分离，避免过拟合。",
     "tags": ["优化器"], "complexity": "complex"},
    {"id": "class:filters.py:MarketRegimeFilter", "type": "class", "name": "MarketRegimeFilter", "filePath": "filters.py",
     "summary": "市场状态识别器。基于ADX/波动率/均线排列判断趋势/震荡/高波动。",
     "tags": ["市场状态"], "complexity": "moderate"},
    {"id": "config:config.py", "type": "config", "name": "回测配置", "filePath": "config.py",
     "summary": "全局配置节点：002036(联创电子)、初始10万、佣金万分之2.5、印花税千分之1。",
     "tags": ["配置"]},
    {"id": "document:requirements.txt", "type": "document", "name": "依赖文档", "filePath": "requirements.txt",
     "summary": "Python依赖声明：akshare>=1.12, pandas>=2.0, numpy>=1.24, matplotlib>=3.7。",
     "tags": ["文档"]},
]

edges = [
    # main.py imports everything
    {"source": "file:main.py", "target": "file:config.py", "type": "imports", "weight": 0.7},
    {"source": "file:main.py", "target": "file:data_loader.py", "type": "imports", "weight": 0.7},
    {"source": "file:main.py", "target": "file:engine.py", "type": "imports", "weight": 0.7},
    {"source": "file:main.py", "target": "file:strategies/__init__.py", "type": "imports", "weight": 0.7},
    {"source": "file:main.py", "target": "file:metrics.py", "type": "imports", "weight": 0.7},
    {"source": "file:main.py", "target": "file:visualize.py", "type": "imports", "weight": 0.7},
    {"source": "file:main.py", "target": "file:optimizer.py", "type": "imports", "weight": 0.7},
    # engine.py
    {"source": "file:engine.py", "target": "file:config.py", "type": "imports", "weight": 0.7},
    {"source": "file:engine.py", "target": "file:strategies/base.py", "type": "imports", "weight": 0.7},
    # data_loader.py
    {"source": "file:data_loader.py", "target": "file:config.py", "type": "imports", "weight": 0.7},
    # optimizer.py
    {"source": "file:optimizer.py", "target": "file:engine.py", "type": "imports", "weight": 0.7},
    {"source": "file:optimizer.py", "target": "file:strategies/__init__.py", "type": "imports", "weight": 0.7},
    # visualize
    {"source": "file:visualize.py", "target": "file:metrics.py", "type": "imports", "weight": 0.7},
    # Strategy base imports
    {"source": "file:strategies/ma_cross.py", "target": "file:strategies/base.py", "type": "imports", "weight": 0.7},
    {"source": "file:strategies/macd.py", "target": "file:strategies/base.py", "type": "imports", "weight": 0.7},
    {"source": "file:strategies/bollinger.py", "target": "file:strategies/base.py", "type": "imports", "weight": 0.7},
    {"source": "file:strategies/rsi.py", "target": "file:strategies/base.py", "type": "imports", "weight": 0.7},
    {"source": "file:strategies/composite.py", "target": "file:strategies/base.py", "type": "imports", "weight": 0.7},
    {"source": "file:strategies/advanced_swing.py", "target": "file:strategies/base.py", "type": "imports", "weight": 0.7},
    {"source": "file:strategies/pattern_swing.py", "target": "file:strategies/base.py", "type": "imports", "weight": 0.7},
    # Registry imports strategies
    {"source": "file:strategies/__init__.py", "target": "file:strategies/ma_cross.py", "type": "imports", "weight": 0.7},
    {"source": "file:strategies/__init__.py", "target": "file:strategies/macd.py", "type": "imports", "weight": 0.7},
    {"source": "file:strategies/__init__.py", "target": "file:strategies/bollinger.py", "type": "imports", "weight": 0.7},
    {"source": "file:strategies/__init__.py", "target": "file:strategies/rsi.py", "type": "imports", "weight": 0.7},
    {"source": "file:strategies/__init__.py", "target": "file:strategies/composite.py", "type": "imports", "weight": 0.7},
    {"source": "file:strategies/__init__.py", "target": "file:strategies/advanced_swing.py", "type": "imports", "weight": 0.7},
    {"source": "file:strategies/__init__.py", "target": "file:strategies/pattern_swing.py", "type": "imports", "weight": 0.7},
    # Contains
    {"source": "file:engine.py", "target": "class:engine.py:BacktestEngine", "type": "contains", "weight": 1.0},
    {"source": "file:strategies/base.py", "target": "class:strategies/base.py:BaseStrategy", "type": "contains", "weight": 1.0},
    {"source": "file:optimizer.py", "target": "class:optimizer.py:WalkForwardOptimizer", "type": "contains", "weight": 1.0},
    {"source": "file:filters.py", "target": "class:filters.py:MarketRegimeFilter", "type": "contains", "weight": 1.0},
    # Calls
    {"source": "file:main.py", "target": "class:engine.py:BacktestEngine", "type": "calls", "weight": 0.8},
    {"source": "class:optimizer.py:WalkForwardOptimizer", "target": "class:engine.py:BacktestEngine", "type": "calls", "weight": 0.8},
    # Inheritance
    {"source": "file:strategies/ma_cross.py", "target": "class:strategies/base.py:BaseStrategy", "type": "inherits", "weight": 0.9},
    {"source": "file:strategies/macd.py", "target": "class:strategies/base.py:BaseStrategy", "type": "inherits", "weight": 0.9},
    {"source": "file:strategies/bollinger.py", "target": "class:strategies/base.py:BaseStrategy", "type": "inherits", "weight": 0.9},
    {"source": "file:strategies/rsi.py", "target": "class:strategies/base.py:BaseStrategy", "type": "inherits", "weight": 0.9},
    {"source": "file:strategies/composite.py", "target": "class:strategies/base.py:BaseStrategy", "type": "inherits", "weight": 0.9},
    {"source": "file:strategies/advanced_swing.py", "target": "class:strategies/base.py:BaseStrategy", "type": "inherits", "weight": 0.9},
    {"source": "file:strategies/pattern_swing.py", "target": "class:strategies/base.py:BaseStrategy", "type": "inherits", "weight": 0.9},
    # Depends
    {"source": "class:engine.py:BacktestEngine", "target": "class:strategies/base.py:BaseStrategy", "type": "depends_on", "weight": 0.6},
    # Config
    {"source": "config:config.py", "target": "file:config.py", "type": "defines_schema", "weight": 0.8},
    # Docs
    {"source": "document:requirements.txt", "target": "file:requirements.txt", "type": "documents", "weight": 0.5},
]

layers = [
    {"id": "layer:entry", "name": "入口层", "description": "CLI入口与流程调度",
     "nodeIds": ["file:main.py"]},
    {"id": "layer:core", "name": "核心引擎层", "description": "回测引擎、全局配置、交易执行",
     "nodeIds": ["file:engine.py", "file:config.py", "class:engine.py:BacktestEngine", "config:config.py"]},
    {"id": "layer:data", "name": "数据层", "description": "行情获取、技术指标、缓存",
     "nodeIds": ["file:data_loader.py"]},
    {"id": "layer:strategy", "name": "策略层", "description": "策略基类与7个具体策略实现",
     "nodeIds": ["file:strategies/__init__.py", "file:strategies/base.py",
                 "file:strategies/ma_cross.py", "file:strategies/macd.py",
                 "file:strategies/bollinger.py", "file:strategies/rsi.py",
                 "file:strategies/composite.py", "file:strategies/advanced_swing.py",
                 "file:strategies/pattern_swing.py", "class:strategies/base.py:BaseStrategy"]},
    {"id": "layer:analysis", "name": "分析优化层", "description": "市场过滤、形态识别、参数优化、绩效、可视化",
     "nodeIds": ["file:filters.py", "file:patterns.py", "file:optimizer.py",
                 "file:metrics.py", "file:visualize.py",
                 "class:filters.py:MarketRegimeFilter", "class:optimizer.py:WalkForwardOptimizer"]},
    {"id": "layer:support", "name": "支撑层", "description": "依赖声明与文档",
     "nodeIds": ["file:requirements.txt", "document:requirements.txt"]},
]

tour = [
    {"order": 1, "title": "项目入口", "description": "main.py 是CLI调度中心，整合所有模块完成策略回测、参数优化和报告生成。",
     "nodeIds": ["file:main.py"]},
    {"order": 2, "title": "全局配置", "description": "config.py 定义股票代码(002036联创电子)、初始资金(10万)、交易成本、策略参数。所有模块依赖此配置。",
     "nodeIds": ["file:config.py", "config:config.py"]},
    {"order": 3, "title": "核心引擎", "description": "engine.py 是系统心脏。BacktestEngine.run() 逐日模拟A股T+1交易，处理止损止盈和成本。",
     "nodeIds": ["file:engine.py", "class:engine.py:BacktestEngine"]},
    {"order": 4, "title": "策略体系", "description": "BaseStrategy 抽象基类 + 7个具体策略(均线/MACD/布林/RSI/复合/高级波段/形态)。注册中心提供工厂方法。",
     "nodeIds": ["file:strategies/base.py", "class:strategies/base.py:BaseStrategy",
                 "file:strategies/__init__.py", "file:strategies/ma_cross.py", "file:strategies/composite.py"]},
    {"order": 5, "title": "优化与分析", "description": "WalkForwardOptimizer避免过拟合，MarketRegimeFilter识别市场状态，metrics.py计算夏普比率和回撤。",
     "nodeIds": ["file:optimizer.py", "class:optimizer.py:WalkForwardOptimizer",
                 "file:filters.py", "file:metrics.py"]},
    {"order": 6, "title": "可视化报告", "description": "visualize.py 生成权益曲线图、交易分析图和完整HTML报告。",
     "nodeIds": ["file:visualize.py"]},
]

graph = {
    "version": "1.0.0",
    "project": {
        "name": PROJECT_NAME,
        "languages": ["python", "html"],
        "frameworks": ["pandas", "numpy", "matplotlib", "scipy", "akshare"],
        "description": PROJECT_DESC,
        "analyzedAt": datetime.now(timezone.utc).isoformat(),
        "gitCommitHash": "no-commits"
    },
    "nodes": nodes,
    "edges": edges,
    "layers": layers,
    "tour": tour
}

with open(f"{PROJECT_ROOT}/.understand-anything/intermediate/assembled-graph.json", "w", encoding="utf-8") as f:
    json.dump(graph, f, ensure_ascii=False, indent=2)

print(f"Graph: {len(nodes)} nodes, {len(edges)} edges, {len(layers)} layers, {len(tour)} tour steps")
