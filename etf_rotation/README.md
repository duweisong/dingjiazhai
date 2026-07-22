# ETF 行业轮动策略

动量+量比双因子 ETF 行业轮动，覆盖 11 大行业 44 只 ETF。

## 策略逻辑

1. 候选池：44 只 A 股行业 ETF
2. 绝对动量过滤：2月动量 > -1.5%
3. 得分配置：2月动量 × 5日量比（得分平方加权）

## 文件

- `etf_rotation.py` — 策略引擎（V4.2 增强版）
- `etf_rotation_live.py` — 实盘信号扫描 + PushPlus 推送

## 用法

```bash
python etf_rotation.py
python etf_rotation_live.py
```
