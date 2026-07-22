// ============================================================
// 掘金日报 Pro - Express 服务器
// 文章第5条: 用Claude Code搭全栈, 你当建筑师它写排水管
// ============================================================

const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

const app = express();
const PORT = process.env.PORT || 3456;

const config = JSON.parse(fs.readFileSync(path.join(__dirname,'..','config.json'),'utf-8'));

app.use(cors());
app.use(express.json());

function log(msg) {
  const ts = new Date().toISOString().slice(0,19).replace('T',' ');
  console.log('['+ts+'] '+msg);
}

// ==================== API: 系统状态 ====================
app.get('/api/status', (req, res) => {
  const latest = getLatestReport();
  const subs = getSubscribers();
  const paying = subs.filter(s=>s.tier!=='free').length;

  res.json({
    ok: true, time: new Date().toISOString(),
    app: config.app,
    stats: {
      stocks_tracked: countStocks(),
      subscribers: subs.length,
      paying_users: paying,
      mrr: calcMRR(subs),
      last_report: latest ? latest.date : null
    }
  });
});

// ==================== API: 每日信号 ====================
app.get('/api/signals', (req, res) => {
  const { uid, tier } = req.query;
  const report = getLatestReport();
  if (!report) return res.json({ok:false, msg:'暂无今日信号, 请等待每日17:00更新'});

  const limit = (tier==='pro'||tier==='vip') ? 30 : 10;

  res.json({
    ok: true, date: report.date,
    top_stocks: report.topStocks.slice(0, limit),
    buy_signals: (tier==='pro'||tier==='vip') ? report.buySignals : 'Pro版解锁查看',
    sell_signals: (tier==='pro'||tier==='vip') ? report.sellSignals : 'Pro版解锁查看',
    portfolio: (tier==='pro'||tier==='vip') ? report.portfolio : [],
    is_premium: (tier==='pro'||tier==='vip'),
    upgrade_url: (tier!=='pro'&&tier!=='vip') ? '/upgrade' : null
  });
});

// ==================== API: 订阅 ====================
app.post('/api/subscribe', (req, res) => {
  const { email, wechat_id, tier } = req.body;
  if (!email && !wechat_id) {
    return res.status(400).json({ok:false, msg:'请提供邮箱或微信号'});
  }

  const subs = getSubscribers();
  const exists = subs.find(s => (email&&s.email===email) || (wechat_id&&s.wechat_id===wechat_id));
  if (exists) {
    return res.json({ok:true, msg:'您已订阅', tier:exists.tier, uid:exists.uid});
  }

  const uid = 'U' + Date.now().toString(36).toUpperCase();
  const t = tier || 'free';
  subs.push({
    uid, email, wechat_id, tier: t,
    created_at: new Date().toISOString(),
    status: 'active'
  });
  saveSubscribers(subs);

  log('新订阅: '+(email||wechat_id)+' tier='+t);
  res.json({
    ok: true,
    msg: t==='free' ? '免费订阅成功!' : '请完成支付激活Pro',
    uid, tier: t,
    payment_url: t!=='free' ? '/upgrade?uid='+uid : null
  });
});

// ==================== API: 支付确认 ====================
app.post('/api/payment/confirm', (req, res) => {
  const { uid, tier, amount } = req.body;
  const subs = getSubscribers();
  const idx = subs.findIndex(s=>s.uid===uid);
  if (idx===-1) return res.status(404).json({ok:false, msg:'用户不存在'});

  const paidUntil = new Date();
  paidUntil.setMonth(paidUntil.getMonth()+1);
  subs[idx].tier = tier;
  subs[idx].paid_until = paidUntil.toISOString();
  subs[idx].amount_paid = amount || (tier==='pro'?29:99);
  saveSubscribers(subs);

  const payments = getPayments();
  payments.push({uid, tier, amount: subs[idx].amount_paid, paid_at: new Date().toISOString()});
  fs.writeFileSync(config.payment_file, JSON.stringify(payments,null,2));

  log('支付确认: '+uid+' '+tier+' Y'+subs[idx].amount_paid);
  res.json({ok:true, msg:'支付成功! 欢迎加入掘金日报Pro', tier, paid_until: paidUntil.toISOString()});
});

// ==================== API: 小红书文案生成 ====================
app.post('/api/redbook/generate', (req, res) => {
  const { prompt, topic, style } = req.body;
  if (!prompt || !topic) {
    return res.status(400).json({ok:false, msg:'缺少话题'});
  }

  // TODO: 接入Claude API实现真正的AI生成
  // 目前返回fallback让前端本地生成
  res.json({ok:true, fallback:true, msg:'使用本地模板生成'});
});

// ==================== API: 管理统计 ====================
app.get('/api/admin/stats', (req, res) => {
  const subs = getSubscribers();
  const payments = getPayments();
  const paying = subs.filter(s=>s.tier!=='free');
  const pro = subs.filter(s=>s.tier==='pro').length;
  const vip = subs.filter(s=>s.tier==='vip').length;
  const free = subs.filter(s=>s.tier==='free').length;
  const mrr = calcMRR(subs);

  res.json({
    total: subs.length,
    paying: paying.length,
    conversion: subs.length>0 ? (paying.length/subs.length*100).toFixed(1)+'%' : '0%',
    mrr, arr: mrr*12,
    by_tier: {free, pro, vip},
    recent_payments: payments.slice(-10).reverse(),
    costs: config.costs
  });
});

// ==================== 页面路由 ====================
app.get('/upgrade', (req,res) => res.sendFile(path.join(__dirname,'..','web','upgrade.html')));
app.get('/admin', (req,res) => res.sendFile(path.join(__dirname,'..','web','admin.html')));
app.get('/redbook', (req,res) => res.sendFile(path.join(__dirname,'..','ideas','redbook-writer','web','index.html')));
app.get('/wechat', (req,res) => res.sendFile(path.join(__dirname,'..','ideas','wechat-writer','web','index.html')));
app.get('/radar', (req,res) => res.sendFile(path.join(__dirname,'..','ideas','review-radar','web','index.html')));

// ==================== Markdown报告解析 ====================
function getLatestReport() {
  const dir = config.signal.reports_dir;
  if (!fs.existsSync(dir)) return null;
  const files = fs.readdirSync(dir)
    .filter(f=>f.startsWith('report_')&&f.endsWith('.md'))
    .sort().reverse();
  if (!files.length) return null;
  return parseReport(fs.readFileSync(path.join(dir,files[0]),'utf-8'));
}

function parseReport(md) {
  const dateMatch = md.match(/\*{0,2}报告日期\*{0,2}[：:]\s*([\d-]+)/);
  const date = dateMatch ? dateMatch[1] : 'unknown';
  const topStocks = [];

  const lines = md.split('\n');
  for (const line of lines) {
    if (line.match(/^\| #\d+/)) {
      const cols = line.split('|').map(c=>c.trim()).filter(Boolean);
      if (cols.length>=6) {
        topStocks.push({
          rank: cols[0].replace('#',''),
          code: cols[1], name: cols[2],
          score: cols[3],
          ret_5d: cols[4], ret_20d: cols[5]
        });
      }
    }
  }

  const buyMatch = md.match(/买入信号[：:]\s*(.+)/);
  const sellMatch = md.match(/卖出信号[：:]\s*(.+)/);

  const portfolio = [];
  let inHold = false;
  for (const line of lines) {
    if (line.includes('当前持仓')) inHold = true;
    else if (inHold && line.match(/^---/)) inHold = false;
    else if (inHold && line.match(/^\| \d{6}/)) {
      const cols = line.split('|').map(c=>c.trim()).filter(Boolean);
      if (cols.length>=6) {
        portfolio.push({
          code: cols[0], name: cols[1],
          buy_price: cols[2], current_price: cols[3],
          pnl: cols[4], hold_days: cols[5]
        });
      }
    }
  }

  return {
    date, topStocks,
    buySignals: buyMatch ? buyMatch[1].trim() : '无',
    sellSignals: sellMatch ? sellMatch[1].trim() : '无',
    portfolio
  };
}

// ==================== 数据管理 ====================
function getSubscribers() {
  try {
    return fs.existsSync(config.subscriber_file)
      ? JSON.parse(fs.readFileSync(config.subscriber_file,'utf-8')) : [];
  } catch(e) { return []; }
}
function saveSubscribers(subs) {
  fs.writeFileSync(config.subscriber_file, JSON.stringify(subs,null,2));
}
function getPayments() {
  try {
    return fs.existsSync(config.payment_file)
      ? JSON.parse(fs.readFileSync(config.payment_file,'utf-8')) : [];
  } catch(e) { return []; }
}
function calcMRR(subs) {
  return subs.reduce((sum,s)=>{
    if (s.tier==='pro') return sum+29;
    if (s.tier==='vip') return sum+99;
    return sum;
  }, 0);
}
function countStocks() {
  try {
    return fs.readdirSync(config.signal.stocks_cache_dir)
      .filter(f=>f.endsWith('.pkl')).length;
  } catch(e) { return 0; }
}

// ==================== 静态文件 (放最后, 不影响路由) ====================
app.use(express.static(path.join(__dirname,'..','web')));
app.use('/ideas', express.static(path.join(__dirname,'..','ideas')));

// ==================== 启动 ====================
app.listen(PORT, () => {
  console.log('');
  console.log('  ========================================');
  console.log('   掘金日报 Pro v1.0');
  console.log('   每天一条信号, 跑赢90%散户');
  console.log('  ========================================');
  console.log('  本地: http://localhost:'+PORT);
  console.log('  管理: http://localhost:'+PORT+'/admin');
  console.log('  Pro Y29/月 | VIP Y99/月');
  console.log('  跟踪 '+countStocks()+' 只A股');
  console.log('');
});
