# 网格交易系统 — 生产部署指南

## 前置条件

### 1. SSL 证书修复（Windows）

efinance/akshare 访问东方财富 API 时可能遇到 SSL 证书验证失败。解决方法（任选其一）：

**方法 A：安装 Windows 证书桥接（推荐）**
```bash
pip install python-certifi-win32
```

**方法 B：手动修复 certifi**
```bash
pip install --upgrade certifi
set SSL_CERT_FILE=C:\Users\<用户名>\AppData\Roaming\Python\Python312\site-packages\certifi\cacert.pem
```

**方法 C：使用系统代理（如果公司网络有代理）**
```bash
set HTTP_PROXY=http://proxy.company.com:8080
set HTTPS_PROXY=http://proxy.company.com:8080
```

### 2. PushPlus Token 配置

1. 访问 https://www.pushplus.plus/ 注册并获取 Token
2. 设置环境变量：
```bash
setx PUSHPLUS_TOKEN "your_token_here"
```

## 部署步骤

### 1. 验证系统

```bash
cd C:\AI\grid-backtest

# 验证测试
python -m pytest tests/ -v

# 生成首次信号
python run_daily_signal.py
```

### 2. 部署定时任务

**自动部署（管理员 PowerShell）：**
```powershell
cd C:\AI\grid-backtest
.\deploy.bat
```

**手动部署：**
```powershell
# 每日信号任务（周一至周五 15:30）
schtasks /create /tn "GridSignal_Daily" `
  /tr "python C:\AI\grid-backtest\run_daily_signal.py --push" `
  /sc weekly /d MON,TUE,WED,THU,FRI /st 15:30

# 查看已部署任务
schtasks /query /fo LIST | findstr Grid

# 手动运行测试
schtasks /run /tn "GridSignal_Daily"
```

### 3. 监控

```bash
# 查看今日日志
type logs\signal_$(date +%Y%m%d).log

# 查看最近信号
python -c "import json; p=json.load(open('position.json')); print(p)"
```

## 日常使用

### 盘后操作（15:30 自动运行）
1. 系统自动从缓存/实时数据获取最新价格
2. 生成每只 ETF 的网格交易信号（BUY/SELL/HOLD/WAIT）
3. 如有高优先级信号 → PushPlus 微信推送
4. 更新持仓记录到 `position.json`

### 周末复盘（周五 16:00 自动运行）
1. 运行全量回测，更新绩效图表
2. 生成 HTML 报告
3. 推送周度总结到微信

## 文件结构

```
grid-backtest/
├── run_daily_signal.py  ← 每日信号生成（定时任务入口）
├── deploy.bat            ← Windows 任务计划部署脚本
├── position.json         ← 当前持仓跟踪
├── logs/                 ← 每日运行日志
├── output/               ← 回测报告和图表
└── .cache/               ← 数据缓存
```

## 故障排查

| 问题 | 解决 |
|------|------|
| SSL 错误 | 运行上述 SSL 修复方法 A |
| 无数据 | 检查 `.cache/` 是否存在 |
| PushPlus 不推送 | 检查 `PUSHPLUS_TOKEN` 环境变量 |
| 定时任务不执行 | `schtasks /query` 查看任务状态 |
| Python 找不到 | 确认 Python 在 PATH 中 |
