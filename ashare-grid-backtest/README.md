# A股网格交易回测系统

基于 entry.py 网格策略逻辑的自包含回测引擎，带 Web 可视化界面。

## 快速开始

```bash
# 1. 安装依赖（只需一次）
pip install pandas numpy matplotlib flask baostock akshare efinance

# 2. 启动 Web 界面
python server.py

# 3. 浏览器打开 http://localhost:8520
```

## 纯命令行模式

```bash
python main.py                          # 默认参数回测
python main.py --symbol 600519 --market SH
python main.py --scan                   # 参数扫描
python main.py --pool stock             # 标的池扫描
python main.py --plot --show            # 生成图表
```

## 自定义端口

```bash
python server.py 9999
```

## 项目结构

```
ashare-grid-backtest/
├── server.py              # Flask Web 服务器
├── main.py                # CLI 入口
├── src/
│   ├── config.py          # 标的池 / 交易成本 / 网格参数
│   ├── data_loader.py     # 数据加载（baostock → akshare → efinance）
│   ├── commissions.py     # A股佣金计算
│   ├── grid_engine.py     # 网格交易回测引擎
│   ├── engine.py          # 回测运行器 + 参数扫描
│   └── visualize.py       # matplotlib 图表
├── design/
│   └── Grid Backtest UI.html  # Web 界面
├── tests/
│   └── test_strategy.py   # 22 个单元测试
└── .cache/                # 16 只标的预缓存数据
```

## 数据源

| 优先级 | 数据源 | 说明 |
|--------|--------|------|
| 1 | 本地缓存 (.cache/) | parquet 格式，秒级加载 |
| 2 | baostock | TCP socket 协议，无 SSL 依赖 |
| 3 | akshare + curl 补丁 | curl TLS 指纹绕过 CDN 检测 |
| 4 | efinance | 东方财富接口 |

## 已缓存标的

16 只标的（2020-2025 日线）：联创电子、贵州茅台、五粮液、宁德时代、中国平安、美的集团、招商银行、海康威视、东方财富、隆基绿能、沪深300ETF、中证500ETF、上证50ETF、创业板ETF、科创50ETF、芯片ETF
