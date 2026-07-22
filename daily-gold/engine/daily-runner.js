// ============================================================
// 掘金日报 Pro - 每日自动执行器
// 功能: 运行Python信号 -> 解析报告 -> 通知订阅者 -> 记录日志
// ============================================================

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const CONFIG = JSON.parse(fs.readFileSync(path.join(__dirname,'..','config.json'),'utf-8'));

async function main() {
  console.log('====================================');
  console.log(' 掘金日报 Pro - 每日执行');
  console.log(' '+new Date().toLocaleString('zh-CN'));
  console.log('====================================\n');

  // Step 1: 运行Python信号脚本
  console.log('[1/4] 运行量化信号引擎...');
  const signalOk = await runPythonSignal();
  if (!signalOk) {
    console.error('[FAIL] 信号生成失败');
    return;
  }

  // Step 2: 读取报告
  console.log('[2/4] 读取最新报告...');
  const report = getLatestReport();
  if (!report) {
    console.error('[FAIL] 无法读取报告');
    return;
  }
  console.log('  报告日期: '+report.date);

  // Step 3: 通知订阅者
  console.log('[3/4] 通知付费用户...');
  const subs = getSubscribers();
  const paying = subs.filter(s => s.tier !== 'free' && s.status === 'active');
  console.log('  付费用户: '+paying.length+' 人');

  // TODO: 集成邮件/微信推送
  for (const user of paying) {
    console.log('  [TODO推送] '+ (user.email || user.wechat_id));
  }

  // Step 4: 保存日志
  console.log('[4/4] 保存运行日志...');
  saveRunLog(report);

  console.log('\n执行完成!');
}

async function runPythonSignal() {
  return new Promise((resolve) => {
    const child = spawn(CONFIG.signal.python_cmd, ['-X', 'utf8', CONFIG.signal.script_path], {
      cwd: 'C:/AI', timeout: 300000
    });

    child.stdout.on('data', (d) => process.stdout.write('  '+d.toString().trim().slice(0,100)+'\n'));
    child.stderr.on('data', (d) => { const m=d.toString().trim(); if(m) console.error('  [err] '+m.slice(0,200)); });
    child.on('close', (code) => resolve(code === 0));
    child.on('error', (err) => { console.error('  [err] '+err.message); resolve(false); });
  });
}

function getLatestReport() {
  const dir = CONFIG.signal.reports_dir;
  if (!fs.existsSync(dir)) return null;
  const files = fs.readdirSync(dir).filter(f=>f.startsWith('report_')&&f.endsWith('.md')).sort().reverse();
  if (!files.length) return null;
  return { date: files[0].replace('report_','').replace('.md',''), file: files[0] };
}

function getSubscribers() {
  try {
    const f = CONFIG.subscriber_file;
    return fs.existsSync(f) ? JSON.parse(fs.readFileSync(f,'utf-8')) : [];
  } catch(e) { return []; }
}

function saveRunLog(report) {
  const logDir = path.join(__dirname,'..','data');
  if (!fs.existsSync(logDir)) fs.mkdirSync(logDir, {recursive:true});
  const entry = { time: new Date().toISOString(), report_date: report.date };
  const logFile = path.join(logDir, 'daily-log.json');
  let logs = [];
  try { logs = JSON.parse(fs.readFileSync(logFile,'utf-8')); } catch(e) {}
  logs.push(entry);
  if (logs.length > 30) logs = logs.slice(-30);
  fs.writeFileSync(logFile, JSON.stringify(logs, null, 2));
}

main().catch(console.error);
