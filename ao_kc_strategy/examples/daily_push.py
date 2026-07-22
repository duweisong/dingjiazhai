#!/usr/bin/env python3
"""AO+KC 每日选股 PushPlus 微信推送"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ao_kc_strategy import *
import requests
from datetime import datetime

TDX = r"C:\zd_pazq_hy"
PP_TOKEN = os.environ.get("PP_TOKEN", "")
PP_URL = "http://www.pushplus.plus/send"

HS300 = [
    "000001","000002","000063","000100","000157","000166","000301","000333","000338",
    "000408","000425","000538","000568","000596","000617","000625","000630","000651",
    "000661","000725","000768","000776","000786","000792","000800","000831","000858",
    "000876","000895","000938","000963","000975","000977","000983","002001","002007",
    "002027","002049","002050","002129","002142","002179","002180","002230","002236",
    "002241","002252","002271","002304","002311","002352","002371","002410","002415",
    "002460","002466","002475","002493","002594","002601","002648","002709","002714",
    "002736","002812","002916","002920","002938","300014","300015","300033","300059",
    "300122","300124","300142","300207","300223","300274","300308","300316","300347",
    "300390","300408","300413","300433","300442","300450","300498","300502","300529",
    "300628","300661","300750","300751","300760","300763","300896","300919","300957",
    "300979","300999","600000","600009","600010","600011","600015","600016","600018",
    "600019","600025","600028","600029","600030","600031","600036","600048","600050",
    "600061","600066","600085","600089","600104","600111","600115","600132","600150",
    "600161","600176","600183","600188","600196","600233","600276","600309","600346",
    "600362","600377","600383","600406","600415","600426","600436","600438","600460",
    "600482","600489","600515","600519","600547","600570","600584","600585","600588",
    "600600","600606","600690","600703","600732","600741","600745","600754","600760",
    "600763","600795","600803","600809","600837","600845","600886","600887","600893",
    "600900","600905","600918","600919","600926","600938","600941","600958","600989",
    "601006","601009","601012","601021","601058","601066","601077","601088",
    "601100","601108","601111","601117","601127","601138","601166","601169","601186",
    "601211","601225","601229","601236","601238","601288","601318","601319","601328",
    "601336","601360","601377","601390","601398","601456","601600","601601","601607",
    "601615","601618","601628","601633","601658","601668","601669","601688","601689",
    "601696","601698","601728","601766","601788","601800","601808","601816","601818",
    "601838","601857","601868","601872","601877","601878","601881","601888","601898",
    "601899","601901","601916","601919","601939","601966","601985","601988","601995",
    "601998","603019","603160","603259","603260","603288","603290","603296","603369",
    "603392","603501","603596","603606","603659","603799","603806","603833","603899",
    "603986","603993","605117","605499","688008","688009","688012","688036","688041",
    "688047","688065","688082","688111","688126","688169","688187","688223","688256",
    "688271","688303","688396","688484","688506","688525","688561","688599","688728",
    "688777","688819","688981",
]

# ---- 股票名称缓存 ----
_NAME_CACHE = {}
_NAME_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".strategy_cache", "stock_names.json")

def _load_name_cache():
    global _NAME_CACHE
    if os.path.exists(_NAME_CACHE_FILE):
        try:
            import json
            with open(_NAME_CACHE_FILE, "r", encoding="utf-8") as f:
                _NAME_CACHE = json.load(f)
        except: pass

def _save_name_cache():
    import json
    os.makedirs(os.path.dirname(_NAME_CACHE_FILE), exist_ok=True)
    with open(_NAME_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(_NAME_CACHE, f, ensure_ascii=False)

def _fetch_names_akshare(codes):
    """从 akshare 批量获取股票名称"""
    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        if df is not None and len(df) > 0:
            for _, row in df.iterrows():
                code = str(row.get("代码", "")).strip()
                name = str(row.get("名称", "")).strip()
                if code and name:
                    _NAME_CACHE[code] = name
            _save_name_cache()
            return True
    except: pass
    return False

def get_name(code):
    """获取股票名称，优先缓存 > akshare > 代码本身"""
    _load_name_cache()
    if code in _NAME_CACHE:
        return f"{code}({_NAME_CACHE[code]})"
    return code

def ensure_names(codes):
    """确保 codes 中所有股票都有名称"""
    _load_name_cache()
    missing = [c for c in codes if c not in _NAME_CACHE]
    if missing:
        _fetch_names_akshare(missing)
    for c in codes:
        if c not in _NAME_CACHE:
            _NAME_CACHE[c] = c

def scan_all(online=False):
    buy, hold, risk = [], [], []
    total = skipped = 0

    # 预先加载名称（在线模式）
    if online:
        ensure_names(HS300)

    for code in HS300:
        try:
            if online:
                df = fetch_akshare(code)
                if df is None or len(df) < 100:
                    skipped += 1; continue
            else:
                fp = find_tdx_file(code, TDX)
                if not fp: skipped += 1; continue
                df = read_tdx_day(fp)
                if len(df) < 100: skipped += 1; continue

            df = calc_ao(df); df = calc_kc(df); df = generate(df)
            L = df.iloc[-1]; total += 1
            name = get_name(code)
            if L["pos"] > 0:
                e, s = L["entry"], L["stop"]
                p = (L["Close"]/e-1)*100; g = (L["Close"]-s)/L["Close"]*100
                hold.append({"c":code,"n":name,"cl":L["Close"],"e":e,"s":s,"p":p,"g":g,"ao":L["ao"]})
                if g < 2 or p < -5:
                    risk.append({"c":code,"n":name,"cl":L["Close"],"e":e,"s":s,"p":p,"g":g})
            elif L["ao_up"] and not L["below_kc"]:
                st = L["Close"] - Cfg.STOP_ATR * L["atr"]
                buy.append({"c":code,"n":name,"cl":L["Close"],"ao":L["ao"],"kc":L["kc_low"],"st":st})
        except: skipped += 1
    buy.sort(key=lambda x: x["ao"], reverse=True)
    hold.sort(key=lambda x: x["p"], reverse=True)
    risk.sort(key=lambda x: x["g"])
    return {"buy":buy,"hold":hold,"risk":risk,"total":total,"skip":skipped,
            "time":datetime.now().strftime("%Y-%m-%d %H:%M")}

def build_html(d):
    css = "body{font-family:sans-serif;background:#f5f5f5;margin:0;padding:10px}" \
          ".card{background:#fff;border-radius:8px;padding:12px;margin-bottom:10px;box-shadow:0 1px 3px rgba(0,0,0,.1)}" \
          "h2{font-size:15px;margin:0 0 8px;border-bottom:2px solid #3498db;padding-bottom:5px}" \
          "h3{font-size:12px;color:#888;margin:3px 0}" \
          "table{width:100%;border-collapse:collapse;font-size:12px}" \
          "th{background:#f0f4f8;padding:5px 8px;text-align:right;border-bottom:1px solid #ddd}" \
          "th:first-child{text-align:center}" \
          "td{padding:5px 8px;text-align:right;border-bottom:1px solid #f0f0f0}" \
          "td:first-child{text-align:center;font-weight:700;font-family:monospace}" \
          ".sum{display:flex;gap:8px}.si{flex:1;text-align:center;padding:8px;background:#f0f4f8;border-radius:6px}" \
          ".sn{font-size:22px;font-weight:700}.sl{font-size:11px;color:#888}" \
          ".tag{display:inline-block;padding:2px 7px;border-radius:8px;font-size:10px;color:#fff;font-weight:600}" \
          ".tbuy{background:#27ae60}.thold{background:#3498db}.trisk{background:#e74c3c}" \
          ".warn{background:#fff3cd;border-left:3px solid #f39c12;padding:6px 10px;font-size:11px;border-radius:4px}" \
          ".ft{text-align:center;color:#bbb;font-size:10px;margin-top:6px}" \
          ".gr{color:#27ae60;font-weight:600}.rd{color:#e74c3c;font-weight:600}"
    row = lambda cells: f"<tr>{''.join(f'<td>{v}</td>' for v in cells)}</tr>"
    thr = lambda cells: f"<tr>{''.join(f'<th>{v}</th>' for v in cells)}</tr>"

    h = f"<!DOCTYPE html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><style>{css}</style></head><body>"
    h += f"<div class='card'><h2>AO+KC Daily Report</h2><h3>{d['time']} | scanned {d['total']} stocks</h3>"
    h += f"<div class='sum'><div class='si'><div class='sn'>{len(d['buy'])}</div><div class='sl'><span class='tag tbuy'>BUY</span></div></div>"
    h += f"<div class='si'><div class='sn'>{len(d['hold'])}</div><div class='sl'><span class='tag thold'>HOLD</span></div></div>"
    h += f"<div class='si'><div class='sn'>{len(d['risk'])}</div><div class='sl'><span class='tag trisk'>RISK</span></div></div></div></div>"

    if d["buy"]:
        h += f"<div class='card'><h2>BUY Signals <span class='tag tbuy'>{len(d['buy'])}</span></h2><table>"
        h += thr(["股票","收盘","AO","KC下轨","止损"])
        for s in d["buy"][:10]:
            h += row([s["n"],f"{s['cl']:.2f}",f"<span class='{'gr' if s['ao']>0 else 'rd'}'>{s['ao']:+.2f}</span>",f"{s['kc']:.2f}",f"{s['st']:.2f}"])
        h += "</table>"
        if any(s["ao"]<0 for s in d["buy"][:10]):
            h += '<div class="warn">部分AO为负值，弱反弹信号，注意控制仓位</div>'
        h += "</div>"

    if d["hold"]:
        h += f"<div class='card'><h2>持仓盈利 TOP10 <span class='tag thold'>{len(d['hold'])}</span></h2><table>"
        h += thr(["股票","成本","现价","浮盈","止损","距止损"])
        for s in d["hold"][:10]:
            cls = "gr" if s["p"]>0 else "rd"
            h += row([s["n"],f"{s['e']:.2f}",f"{s['cl']:.2f}",f"<span class='{cls}'>{s['p']:+.1f}%</span>",f"{s['s']:.2f}",f"{s['g']:.1f}%"])
        h += "</table></div>"

    if d["risk"]:
        h += f"<div class='card'><h2>风险预警 <span class='tag trisk'>{len(d['risk'])}</span></h2><table>"
        h += thr(["股票","成本","现价","浮亏","止损","距离"])
        for s in d["risk"][:8]:
            h += row([s["n"],f"{s['e']:.2f}",f"{s['cl']:.2f}",f"<span class='rd'>{s['p']:.1f}%</span>",f"<span class='rd'>{s['s']:.2f}</span>",f"<span class='rd'>{s['g']:.1f}%</span>"])
        h += "</table><div class='warn'>距止损不足2%或浮亏超5%，密切关注开盘</div></div>"

    h += f'<div class="ft">AO+KC Strategy | Auto {d["time"]}</div></body></html>'
    return h

def push(title, content):
    resp = requests.post(PP_URL, json={"token":PP_TOKEN,"title":title,"content":content,"template":"html"}, timeout=15)
    r = resp.json()
    ok = r.get("code") == 200
    print(f"  PushPlus: {'OK' if ok else r}")
    return ok

def run_etf_rotation():
    """运行ETF轮动策略并返回结果文本"""
    import subprocess
    etf_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "etf_rotation")
    try:
        result = subprocess.run(
            ["python3", "-X", "utf8", "etf_rotation_live.py"],
            cwd=etf_dir, capture_output=True, text=True, timeout=120,
            encoding="utf-8", errors="replace"
        )
        lines = (result.stdout or "").split("\n")
        out = []
        capture = False
        for line in lines:
            if "建议持仓" in line: capture = True
            if capture and line.strip(): out.append(line.strip())
            if "交易信号" in line: capture = True
            if "PushPlus" in line: capture = False
            if len(out) > 15: break
        return "\n".join(out) if out else "ETF Rotation: no signal"
    except Exception as e:
        return f"ETF Rotation: {e}"

def build_combined_html(ao_data, etf_text):
    """合并AO+KC和ETF轮动的HTML"""
    css = "body{font-family:sans-serif;background:#f5f5f5;margin:0;padding:10px}" \
          ".card{background:#fff;border-radius:8px;padding:12px;margin-bottom:10px;box-shadow:0 1px 3px rgba(0,0,0,.1)}" \
          "h2{font-size:15px;margin:0 0 8px;border-bottom:2px solid #3498db;padding-bottom:5px}" \
          "h3{font-size:12px;color:#888;margin:3px 0}" \
          "table{width:100%;border-collapse:collapse;font-size:12px}" \
          "th{background:#f0f4f8;padding:5px 8px;text-align:right;border-bottom:1px solid #ddd}" \
          "th:first-child{text-align:center}" \
          "td{padding:5px 8px;text-align:right;border-bottom:1px solid #f0f0f0}" \
          "td:first-child{text-align:center;font-weight:700;font-family:monospace}" \
          ".sum{display:flex;gap:8px}.si{flex:1;text-align:center;padding:8px;background:#f0f4f8;border-radius:6px}" \
          ".sn{font-size:22px;font-weight:700}.sl{font-size:11px;color:#888}" \
          ".tag{display:inline-block;padding:2px 7px;border-radius:8px;font-size:10px;color:#fff;font-weight:600}" \
          ".tbuy{background:#27ae60}.thold{background:#3498db}.trisk{background:#e74c3c}.tetf{background:#9b59b6}" \
          ".warn{background:#fff3cd;border-left:3px solid #f39c12;padding:6px 10px;font-size:11px;border-radius:4px}" \
          ".ft{text-align:center;color:#bbb;font-size:10px;margin-top:6px}" \
          ".gr{color:#27ae60;font-weight:600}.rd{color:#e74c3c;font-weight:600}" \
          "pre{background:#f8f9fa;padding:8px;border-radius:4px;font-size:11px;white-space:pre-wrap}"
    row = lambda cells: f"<tr>{''.join(f'<td>{v}</td>' for v in cells)}</tr>"
    thr = lambda cells: f"<tr>{''.join(f'<th>{v}</th>' for v in cells)}</tr>"

    d = ao_data
    h = f"<!DOCTYPE html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><style>{css}</style></head><body>"
    h += f"<div class='card' style='background:linear-gradient(135deg,#1a1a2e,#16213e);color:#fff'>"
    h += f"<h2 style='border-color:#3498db'>Daily Digest</h2>"
    h += f"<h3 style='color:#aaa'>{d['time']}</h3></div>"

    # ---- AO+KC ----
    h += f"<div class='card'><h2>AO+KC <span class='tag tbuy'>{len(d['buy'])}</span> <span class='tag thold'>{len(d['hold'])}</span> <span class='tag trisk'>{len(d['risk'])}</span></h2>"
    if d["buy"]:
        h += f"<h3>BUY Signals</h3><table>"
        h += thr(["Code","Close","AO","KC Low","Stop"])
        for s in d["buy"][:8]:
            cls = "gr" if s["ao"]>0 else "rd"
            h += row([s["c"],f"${s['cl']:.2f}",f"<span class='{cls}'>{s['ao']:+.2f}</span>",f"${s['kc']:.2f}",f"${s['st']:.2f}"])
        h += "</table>"
    if d["risk"]:
        h += f"<h3>RISK Alert</h3><table>"
        h += thr(["Code","Entry","Close","Loss","Stop","Gap"])
        for s in d["risk"][:5]:
            h += row([s["c"],f"${s['e']:.2f}",f"${s['cl']:.2f}",f"<span class='rd'>{s['p']:.1f}%</span>",f"<span class='rd'>${s['s']:.2f}</span>",f"<span class='rd'>{s['g']:.1f}%</span>"])
        h += "</table>"
    h += "</div>"

    # ---- ETF ----
    if etf_text:
        h += f"<div class='card'><h2>ETF Rotation <span class='tag tetf'>ETF</span></h2><pre>{etf_text}</pre></div>"

    h += f'<div class="ft">Auto-generated {d["time"]}</div></body></html>'
    return h


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ao-only", action="store_true", help="Only AO+KC")
    p.add_argument("--etf-only", action="store_true", help="Only ETF rotation")
    p.add_argument("--online", action="store_true", help="Use akshare online data")
    args = p.parse_args()

    # ETF
    etf_text = ""
    if not args.ao_only:
        print("Running ETF Rotation...")
        etf_text = run_etf_rotation()
        print(f"  ETF OK ({len(etf_text)} chars)")

    # AO+KC
    if not args.etf_only:
        src = "akshare" if args.online else "TDX"
        print(f"Scanning CSI300 via {src}...")
        d = scan_all(online=args.online)
        print(f"  Buy:{len(d['buy'])} Hold:{len(d['hold'])} Risk:{len(d['risk'])}")
    else:
        d = {"buy":[],"hold":[],"risk":[],"total":0,"skip":0,"time":datetime.now().strftime("%Y-%m-%d %H:%M")}

    html = build_combined_html(d, etf_text)
    title = f"Daily AO+KC B{len(d['buy'])}R{len(d['risk'])}"
    push(title, html)

    with open("c:/AI/ao_kc_strategy/results/latest_report.html","w",encoding="utf-8") as f:
        f.write(html)
    print("  Done! Check WeChat.")
