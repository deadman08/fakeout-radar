import json, os, re, time
from datetime import datetime, timezone
import pandas as pd
import requests
import yfinance as yf

UA='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131 Safari/537.36'

NIFTY_URL='https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv'
NIFTY50_URL='https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv'

def get_nifty(url):
    r=requests.get(url,headers={'User-Agent':UA},timeout=30)
    r.raise_for_status()
    from io import StringIO
    df=pd.read_csv(StringIO(r.text))
    sym_col=next(c for c in df.columns if 'symbol' in c.lower())
    name_col=next((c for c in df.columns if 'company' in c.lower() or 'name' in c.lower()),sym_col)
    return [(str(s).strip(),str(n).strip()) for s,n in zip(df[sym_col],df[name_col])]

def get_sp500():
    # Wikipedia blocks urllib's default client on GitHub runners; fetch explicitly with a browser-like header.
    url='https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    r=requests.get(url,headers={'User-Agent':UA,'Accept':'text/html,application/xhtml+xml'},timeout=30)
    r.raise_for_status()
    from io import StringIO
    tables=pd.read_html(StringIO(r.text))
    df=tables[0]
    return [(str(s).strip(),str(n).strip()) for s,n in zip(df['Symbol'],df['Security'])]

def normalize(sym,market):
    if market in ('nifty50','nifty500'):
        return sym.replace('&','-')+'.NS'
    return sym.replace('.','-')

def week_bounds(now):
    d=now.normalize()
    mon=d-pd.Timedelta(days=d.weekday())
    prev=mon-pd.Timedelta(days=7)
    return prev,mon

def scan_market(items,market):
    now=pd.Timestamp.now(tz='Asia/Kolkata').tz_localize(None)
    prev_mon,cur_mon=week_bounds(now)
    prev_end=cur_mon-pd.Timedelta(days=1)
    start=prev_mon-pd.Timedelta(days=5)
    # We use period labels from exchange-local timestamps after removing timezone.
    rows=[]
    for j in range(0,len(items),60):
        batch=items[j:j+60]
        tickers=[normalize(s,market) for s,_ in batch]
        try:
            raw=yf.download(tickers,start=start.strftime('%Y-%m-%d'),end=(now+pd.Timedelta(days=1)).strftime('%Y-%m-%d'),interval='1d',auto_adjust=False,group_by='ticker',threads=True,progress=False)
        except Exception:
            raw=pd.DataFrame()
        for sym,name in batch:
            t=normalize(sym,market)
            try:
                if len(batch)==1:
                    x=raw.copy()
                else:
                    x=raw[t].copy()
                x=x.dropna(subset=['High','Low','Close'])
                if x.empty: continue
                if isinstance(x.index,pd.DatetimeIndex) and x.index.tz is not None: x.index=x.index.tz_localize(None)
                prev=x[(x.index>=prev_mon)&(x.index<cur_mon)]
                cur=x[x.index>=cur_mon]
                if prev.empty or cur.empty: continue
                ph=float(prev['High'].max()); pl=float(prev['Low'].min())
                cur=cur.sort_index()
                cw_high=float(cur['High'].max()); cw_low=float(cur['Low'].min()); latest=float(cur['Close'].iloc[-1])
                high_trigger=cur[(cur['High']>ph)&(cur['Close']<ph)]
                low_trigger=cur[(cur['Low']<pl)&(cur['Close']>pl)]
                candidates=[]
                if not high_trigger.empty:
                    r=high_trigger.iloc[0]; trigger=high_trigger.index[0]
                    sweep=float(r['High']); close=float(r['Close']); vol=(cw_high-cw_low)/close*100; dist=(ph-close)/ph*100
                    candidates.append(('HIGH',trigger,sweep,close,vol,dist))
                if not low_trigger.empty:
                    r=low_trigger.iloc[0]; trigger=low_trigger.index[0]
                    sweep=float(r['Low']); close=float(r['Close']); vol=(cw_high-cw_low)/close*100; dist=(close-pl)/pl*100
                    candidates.append(('LOW',trigger,sweep,close,vol,dist))
                for signal,trigger,sweep,close,vol,dist in candidates:
                    rows.append({'symbol':sym,'company':name,'signal':signal,'trigger_date':pd.Timestamp(trigger).strftime('%Y-%m-%d'),'prev_week_high':round(ph,4),'prev_week_low':round(pl,4),'sweep_price':round(sweep,4),'trigger_close':round(close,4),'volatility_pct':round(vol,2),'distance_pct':round(dist,2)})
            except Exception:
                continue
    return {'generated_at':datetime.now(timezone.utc).isoformat(),'data_as_of':now.strftime('%Y-%m-%d %H:%M'),'week_start':cur_mon.strftime('%Y-%m-%d'),'week_end':(cur_mon+pd.Timedelta(days=4)).strftime('%Y-%m-%d'),'source':'Yahoo Finance daily OHLC; membership from Nifty Indices / Wikipedia S&P 500','rows':rows,'counts':{'all':len(rows),'high':sum(r['signal']=='HIGH' for r in rows),'low':sum(r['signal']=='LOW' for r in rows)}}

def main():
    os.makedirs('site/data',exist_ok=True)
    markets={'nifty50':get_nifty(NIFTY50_URL),'nifty500':get_nifty(NIFTY_URL),'us500':get_sp500()}
    for k,v in markets.items():
        data=scan_market(v,k)
        with open(f'site/data/{k}.json','w',encoding='utf-8') as f: json.dump(data,f,ensure_ascii=False,indent=2)
        print(k,len(v),'members',len(data['rows']),'fakeouts')

if __name__=='__main__': main()
