import json, os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf
from scan import get_nifty, NIFTY50_URL

TIMEFRAMES = {
    "5m": ("5m", "5d", None),
    "15m": ("15m", "20d", None),
    "30m": ("30m", "30d", None),
    "1H": ("60m", "60d", None),
    "4H": ("60m", "60d", "4h"),
    "1D": ("1d", "2y", None),
    "1W": ("1d", "2y", "1w"),
}

def _download(tickers, interval, period):
    try:
        return yf.download(tickers, period=period, interval=interval, auto_adjust=False,
                            group_by="ticker", threads=True, progress=False, prepost=False)
    except Exception:
        return pd.DataFrame()

def _one(raw, ticker, many):
    try:
        x = raw[ticker].copy() if many else raw.copy()
    except Exception:
        return pd.DataFrame()
    if x.empty:
        return x
    x.columns = [str(c).title() for c in x.columns]
    x = x[[c for c in ["Open","High","Low","Close","Volume"] if c in x.columns]].copy()
    x = x.dropna(subset=["High","Low","Close"])
    if isinstance(x.index, pd.DatetimeIndex) and x.index.tz is not None:
        x.index = x.index.tz_convert("Asia/Kolkata").tz_localize(None)
    return x.sort_index()

def _resample(x, rule):
    if x.empty: return x
    return x.resample(rule).agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna(subset=["Open","High","Low","Close"])

def _pivots(x, n=3):
    hi=[]; lo=[]; h=x["High"].to_numpy(); l=x["Low"].to_numpy()
    for i in range(n, len(x)-n):
        if h[i] >= max(h[i-n:i+n+1]): hi.append(i)
        if l[i] <= min(l[i-n:i+n+1]): lo.append(i)
    return hi,lo

def _rsi(close,n=14):
    d=close.diff()
    up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean()
    dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    rs=up/dn.replace(0,np.nan)
    return 100-(100/(1+rs))

def _detect(x,side):
    if len(x)<80: return None
    x=x.copy()
    x["ema20"]=x.Close.ewm(span=20,adjust=False).mean()
    x["ema50"]=x.Close.ewm(span=50,adjust=False).mean()
    x["rsi"]=_rsi(x.Close)
    x["vma20"]=x.Volume.rolling(20).mean()
    hi,lo=_pivots(x)
    price=float(x.Close.iloc[-1])
    vr=float(x.Volume.iloc[-1]/x.vma20.iloc[-1]) if pd.notna(x.vma20.iloc[-1]) and x.vma20.iloc[-1]>0 else 1
    best=None
    if side=="BUY":
        for i1 in lo[-10:]:
            hs=[i for i in hi if i>i1]
            if not hs: continue
            i2=max(hs,key=lambda i:x.High.iloc[i])
            ls=[i for i in lo if i>i2]
            if not ls: continue
            i3=ls[-1]; p1=float(x.Low.iloc[i1]); p2=float(x.High.iloc[i2]); p3=float(x.Low.iloc[i3])
            if p3<=p1 or p2<=p1: continue
            retr=(p2-p3)/(p2-p1)
            if not .30<=retr<=.80: continue
            if price>=p2: setup="Wave 3 breakout"
            elif .38<=retr<=.66 and price>=p3: setup="Wave 2 → 3 preparing"
            else: continue
            score=sum([float(x.ema20.iloc[-1])>float(x.ema50.iloc[-1]),float(x.rsi.iloc[-1])>=55,vr>=1.05,price>=p2*.985,p3>p1])
            if score<3: continue
            cand=(score,setup,i1,i2,i3,p1,p2,p3,retr)
            if best is None or cand[0]>best[0]: best=cand
    else:
        for i1 in hi[-10:]:
            ls=[i for i in lo if i>i1]
            if not ls: continue
            i2=min(ls,key=lambda i:x.Low.iloc[i])
            hs=[i for i in hi if i>i2]
            if not hs: continue
            i3=hs[-1]; p1=float(x.High.iloc[i1]); p2=float(x.Low.iloc[i2]); p3=float(x.High.iloc[i3])
            if p3>=p1 or p1<=p2: continue
            retr=(p3-p2)/(p1-p2)
            if not .30<=retr<=.80: continue
            if price<=p2: setup="Wave 3 breakdown"
            elif .38<=retr<=.66 and price<=p3: setup="Wave 2 → 3 preparing"
            else: continue
            score=sum([float(x.ema20.iloc[-1])<float(x.ema50.iloc[-1]),float(x.rsi.iloc[-1])<=45,vr>=1.05,price<=p2*1.015,p3<p1])
            if score<3: continue
            cand=(score,setup,i1,i2,i3,p1,p2,p3,retr)
            if best is None or cand[0]>best[0]: best=cand
    if not best: return None
    score,setup,i1,i2,i3,p1,p2,p3,retr=best
    if side=="BUY":
        fib={"0":p2,"38.2":p2-(p2-p1)*.382,"50":p2-(p2-p1)*.5,"61.8":p2-(p2-p1)*.618,"78.6":p2-(p2-p1)*.786,"100":p1}
        wave_len=p2-p1
        targets={"T1":p3+wave_len*0.618,"T2":p3+wave_len*1.0,"T3":p3+wave_len*1.618}
        invalidation=p1
    else:
        fib={"0":p2,"38.2":p2+(p1-p2)*.382,"50":p2+(p1-p2)*.5,"61.8":p2+(p1-p2)*.618,"78.6":p2+(p1-p2)*.786,"100":p1}
        wave_len=p1-p2
        targets={"T1":p3-wave_len*0.618,"T2":p3-wave_len*1.0,"T3":p3-wave_len*1.618}
        invalidation=p1
    return {
        "score":int(score),"setup":setup,"price":round(price,2),"trigger":round(p2,2),
        "rsi":round(float(x.rsi.iloc[-1]),1),"volume_ratio":round(vr,2),"retracement_pct":round(retr*100,1),
        "pivots":[{"label":"0","time":x.index[i1].isoformat(),"price":round(p1,2)},
                  {"label":"1","time":x.index[i2].isoformat(),"price":round(p2,2)},
                  {"label":"2","time":x.index[i3].isoformat(),"price":round(p3,2)}],
        "fib":{k:round(v,2) for k,v in fib.items()},
        "targets":{k:round(v,2) for k,v in targets.items()},"invalidation":round(invalidation,2),
        "chart":[{"time":int(ts.timestamp()),"open":round(float(r.Open),2),"high":round(float(r.High),2),
                  "low":round(float(r.Low),2),"close":round(float(r.Close),2),"volume":int(r.Volume) if pd.notna(r.Volume) else 0}
                 for ts,r in x.tail(180).iterrows()]
    }

def main():
    nifty50=get_nifty(NIFTY50_URL)
    tickers=[s+".NS" for s,_ in nifty50]
    raws={}
    for tf,(interval,period,rule) in TIMEFRAMES.items():
        key=interval+"|"+period
        if key not in raws: raws[key]=_download(tickers,interval,period)
    buy=[]; sell=[]; details={}
    for sym,name in nifty50:
        details[sym]={"symbol":sym,"company":name,"timeframes":{}}
        best_b=None; best_s=None
        for tf,(interval,period,rule) in TIMEFRAMES.items():
            x=_one(raws.get(interval+"|"+period,pd.DataFrame()),sym+".NS",True)
            if rule: x=_resample(x,rule)
            b=_detect(x,"BUY"); s=_detect(x,"SELL")
            if b:
                details[sym]["timeframes"].setdefault(tf,{})["buy"]=b
                if best_b is None or b["score"]>best_b[1]["score"]: best_b=(tf,b)
            if s:
                details[sym]["timeframes"].setdefault(tf,{})["sell"]=s
                if best_s is None or s["score"]>best_s[1]["score"]: best_s=(tf,s)
        if best_b:
            tf,b=best_b; buy.append({"symbol":sym,"company":name,"timeframe":tf,**{k:b[k] for k in ["setup","price","trigger","score","rsi","volume_ratio","retracement_pct","targets","invalidation"]},"t1":b["targets"]["T1"],"t2":b["targets"]["T2"],"t3":b["targets"]["T3"]})
        if best_s:
            tf,s=best_s; sell.append({"symbol":sym,"company":name,"timeframe":tf,**{k:s[k] for k in ["setup","price","trigger","score","rsi","volume_ratio","retracement_pct","targets","invalidation"]},"t1":s["targets"]["T1"],"t2":s["targets"]["T2"],"t3":s["targets"]["T3"]})
    out={"generated_at":datetime.now(timezone.utc).isoformat(),
         "source":"Yahoo Finance intraday OHLC; NIFTY 50 constituents from NSE",
         "note":"Rule-based Elliott-style heuristic. Wave labels are potential structures, not definitive Elliott Wave analysis.",
         "timeframes":list(TIMEFRAMES.keys()),"buy":sorted(buy,key=lambda r:(-r["score"],r["symbol"])),
         "sell":sorted(sell,key=lambda r:(-r["score"],r["symbol"])),"details":details}
    os.makedirs("site/data",exist_ok=True)
    with open("site/data/elliott.json","w",encoding="utf-8") as f: json.dump(out,f,ensure_ascii=False,separators=(",",":"))
    print("Elliott BUY:",len(buy),"SELL:",len(sell))

if __name__=="__main__": main()
