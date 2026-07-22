# 掘金日报 Pro - 每天自动赚钱的微SaaS系统

> 按 Claude Code 微SaaS爆款文章 10 条方法论构建  
> "代码只花了一个周末。客户花了一年。"

---

## 快速开始

```bash
cd C:\AI\daily-gold
npm install        # 首次运行 (已完成)
npm start          # 启动服务器
```

或双击 `start.bat`

浏览器打开:
- **首页**: http://localhost:3456
- **管理后台**: http://localhost:3456/admin
- **升级页**: http://localhost:3456/upgrade

---

## 系统架构

```
daily-gold/
├── server/
│   └── server.js          # Express API服务器 (核心)
├── engine/
│   └── daily-runner.js    # 每日自动执行器
├── web/
│   ├── index.html         # 营销落地页 (单页验证)
│   ├── upgrade.html       # 支付升级页
│   └── admin.html         # 管理后台
├── data/
│   ├── subscribers.json   # 订阅用户数据库
│   └── payments.json      # 支付记录
├── config.json            # 系统配置
├── package.json
└── start.bat              # 一键启动
```

---

## 商业模式 (按文章方法论)

| # | 文章原则 | 掘金日报对应 |
|---|---------|-------------|
| 1 | 简单数学 | Y29 x 1000用户 = Y29,000/月 |
| 2 | 痛点挖掘 | 散户需要专业量化信号但买不起万元系统 |
| 3 | 先验证 | 单页落地页 + 免费订阅收邮箱 |
| 4 | 便宜栈 | Node.js + HTML/CSS = 月成本 ~Y5 (仅AI API) |
| 5 | Claude Code搭建 | 整个系统由Claude Code一次性生成 |
| 6 | 前10个客户 | 从股吧/雪球/微信群手动推广 |
| 7 | 找下一千个 | 内容营销 + 口碑传播 |
| 8 | 定价复利 | 单档位Y29起步, 付费后再涨价 |
| 9 | 聊天式开发 | 自然语言需求 -> 完整产品 |
| 10 | 坚持 | 代码一周末, 客户一年 |

---

## API 接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/status` | GET | 系统状态 (股票数/用户/MRR) |
| `/api/signals?tier=free` | GET | 免费版信号 (Top10) |
| `/api/signals?tier=pro` | GET | Pro版信号 (Top30+买卖点) |
| `/api/subscribe` | POST | 订阅 (body: email, tier) |
| `/api/payment/confirm` | POST | 支付确认 (body: uid, tier, amount) |
| `/api/admin/stats` | GET | 管理统计 |
| `/api/run-daily` | POST | 手动触发每日信号 |

---

## 定价

| 档位 | 月价 | 年价 | 功能 |
|------|------|------|------|
| 免费 | Y0 | Y0 | Top10信号, 市场概况 |
| **Pro** | **Y29** | **Y290** | Top30+买卖信号+风险监控+邮件推送 |
| VIP | Y99 | Y990 | Pro全部+AI增强+微信推送+一对一咨询 |

---

## 每日执行

```bash
# 手动执行
npm run daily

# 或直接调Python
python -X utf8 C:\AI\daily_signal.py
```

---

## 待办/TODO

- [ ] 微信支付真实对接
- [ ] 邮件推送 (Resend/QQ邮箱)
- [ ] 微信机器人推送
- [ ] AI增强报告 (接入Claude API分析)
- [ ] 用户Dashboard独立页面
- [ ] 部署到Vercel/阿里云

---

*本系统仅供学习研究, 不构成投资建议。股市有风险, 投资需谨慎。*
