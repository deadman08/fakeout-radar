import json, os, re, time, zipfile
from io import BytesIO, StringIO
from datetime import datetime, timezone
import pandas as pd
import requests
import yfinance as yf

UA='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131 Safari/537.36'

NIFTY_URL='https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv'
NIFTY50_URL='https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv'
NSE_BHAVCOPY='https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{date}_F_0000.csv.zip'


def get_nifty(url):
    r=requests.get(url,headers={'User-Agent':UA},timeout=30)
    r.raise_for_status()
    df=pd.read_csv(StringIO(r.text))
    sym_col=next(c for c in df.columns if 'symbol' in c.lower())
    name_col=next((c for c in df.columns if 'company' in c.lower() or 'name' in c.lower()),sym_col)
    return [(str(s).strip(),str(n).strip()) for s,n in zip(df[sym_col],df[name_col])]


def get_sp500():
    url='https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    r=requests.get(url,headers={'User-Agent':UA,'Accept':'text/html,application/xhtml+xml'},timeout=30)
    r.raise_for_status()
    tables=pd.read_html(StringIO(r.text))
    df=tables[0]
    return [(str(s).strip(),str(n).strip()) for s,n in zip(df['Symbol'],df['Security'])]


def normalize(sym,market):
    if market in ('nifty50','nifty500'):
        # India is NSE-only. The .NS suffix also targets NSE on Yahoo when used elsewhere.
        return sym.replace('&','-')+'.NS'
    return sym.replace('.','-')


def week_bounds(now):
    d=now.normalize()
    mon=d-pd.Timedelta(days=d.weekday())
    prev=mon-pd.Timedelta(days=7)
    return prev,mon


def get_nse_bhavcopy(trade_date, wanted_symbols, session):
    date_str=pd.Timestamp(trade_date).strftime('%Y%m%d')
    url=NSE_BHAVCOPY.format(date=date_str)
    try:
        r=session.get(
            url,
            headers={
                'User-Agent':UA,
                'Accept':'application/zip,application/octet-stream,*/*',
                'Referer':'https://www.nseindia.com/all-reports',
            },
            timeout=30,
        )
        if r.status_code != 200 or not r.content.startswith(b'PK'):
            return pd.DataFrame(columns=['Symbol','High','Low','Close'], index=pd.DatetimeIndex([],name='Date'))
        with zipfile.ZipFile(BytesIO(r.content)) as z:
            csv_name=next((n for n in z.namelist() if n.lower().endswith('.csv')),None)
            if not csv_name:
                return pd.DataFrame(columns=['Symbol','High','Low','Close'], index=pd.DatetimeIndex([],name='Date'))
            df=pd.read_csv(z.open(csv_name))
        df.columns=[str(c).strip() for c in df.columns]
        required={'TckrSymb','SctySrs','HghPric','LwPric','ClsPric'}
        if not required.issubset(df.columns):
            return pd.DataFrame(columns=['Symbol','High','Low','Close'], index=pd.DatetimeIndex([],name='Date'))
        df['TckrSymb']=df['TckrSymb'].astype(str).str.strip()
        df=df[df['TckrSymb'].isin(wanted_symbols)]
        if df.empty:
            return pd.DataFrame(columns=['Symbol','High','Low','Close'], index=pd.DatetimeIndex([],name='Date'))
        # Prefer normal equity series where available.
        eq=df[df['SctySrs'].astype(str).str.upper().eq('EQ')]
        if not eq.empty:
            df=eq
        for c in ['HghPric','LwPric','ClsPric']:
            df[c]=pd.to_numeric(df[c],errors='coerce')
        df=df.dropna(subset=['HghPric','LwPric','ClsPric'])
        if df.empty:
            return pd.DataFrame(columns=['Symbol','High','Low','Close'], index=pd.DatetimeIndex([],name='Date'))
        df['Date']=pd.Timestamp(trade_date).normalize()
        return df[['Date','TckrSymb','HghPric','LwPric','ClsPric']].rename(
            columns={'TckrSymb':'Symbol','HghPric':'High','LwPric':'Low','ClsPric':'Close'}
        ).set_index('Date')
    except Exception:
        return pd.DataFrame(columns=['Symbol','High','Low','Close'], index=pd.DatetimeIndex([],name='Date'))


def load_nse_ohlc(symbols, prev_mon, now):
    # Pull NSE's official CM bhavcopy for each calendar day in the prior+current week.
    # Only completed exchange days have a file, so weekends/holidays are naturally skipped.
    session=requests.Session()
    wanted=set(str(s).strip() for s in symbols)
    chunks=[]
    for d in pd.date_range(prev_mon, now.normalize(), freq='D'):
        if d.weekday() >= 5:
            continue
        part=get_nse_bhavcopy(d,wanted,session)
        if not part.empty:
            chunks.append(part)
        time.sleep(0.08)
    if not chunks:
        return {}
    all_df=pd.concat(chunks).sort_index()
    out={}
    for sym, g in all_df.groupby('Symbol'):
        x=g[['High','Low','Close']].copy().sort_index()
        x=x[~x.index.duplicated(keep='last')]
        out[str(sym)]=x
    return out


def scan_market(items,market,nse_cache=None):
    now=pd.Timestamp.now(tz='Asia/Kolkata').tz_localize(None)
    prev_mon,cur_mon=week_bounds(now)
    start=prev_mon-pd.Timedelta(days=5)
    rows=[]

    # US data continues to use Yahoo; Indian data comes from NSE bhavcopy only.
    for j in range(0,len(items),60):
        batch=items[j:j+60]

        if market in ('nifty50','nifty500'):
            raw=None
        else:
            tickers=[normalize(s,market) for s,_ in batch]
            try:
                raw=yf.download(
                    tickers,
                    start=start.strftime('%Y-%m-%d'),
                    end=(now+pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
                    interval='1d',
                    auto_adjust=False,
                    group_by='ticker',
                    threads=True,
                    progress=False
                )
            except Exception:
                raw=pd.DataFrame()

        for sym,name in batch:
            try:
                if market in ('nifty50','nifty500'):
                    x=(nse_cache or {}).get(str(sym).strip(),pd.DataFrame()).copy()
                else:
                    t=normalize(sym,market)
                    if len(batch)==1:
                        x=raw.copy()
                    else:
                        x=raw[t].copy()

                x=x.dropna(subset=['High','Low','Close'])
                if x.empty:
                    continue
                if isinstance(x.index,pd.DatetimeIndex) and x.index.tz is not None:
                    x.index=x.index.tz_localize(None)

                prev=x[(x.index>=prev_mon)&(x.index<cur_mon)]
                cur=x[x.index>=cur_mon]
                if prev.empty or cur.empty:
                    continue

                ph=float(prev['High'].max())
                pl=float(prev['Low'].min())
                cur=cur.sort_index()

                cw_high=float(cur['High'].max())
                cw_low=float(cur['Low'].min())
                latest=float(cur['Close'].iloc[-1])

                # Qualify only when:
                # 1) current week swept the prior week's high/low,
                # 2) the sweep candle closed back inside the prior-week range,
                # 3) latest completed close is still inside that range.
                inside_now=pl < latest < ph
                high_trigger=cur[(cur['High']>ph)&(cur['Close']<ph)]
                low_trigger=cur[(cur['Low']<pl)&(cur['Close']>pl)]

                candidates=[]
                if inside_now and not high_trigger.empty:
                    r=high_trigger.iloc[0]
                    trigger=high_trigger.index[0]
                    sweep=float(r['High'])
                    close=float(r['Close'])
                    vol=(cw_high-cw_low)/close*100
                    dist=(ph-latest)/ph*100
                    candidates.append(('HIGH',trigger,sweep,close,vol,dist,latest))

                if inside_now and not low_trigger.empty:
                    r=low_trigger.iloc[0]
                    trigger=low_trigger.index[0]
                    sweep=float(r['Low'])
                    close=float(r['Close'])
                    vol=(cw_high-cw_low)/close*100
                    dist=(latest-pl)/pl*100
                    candidates.append(('LOW',trigger,sweep,close,vol,dist,latest))

                for signal,trigger,sweep,close,vol,dist,current_close in candidates:
                    rows.append({
                        'symbol':sym,
                        'company':name,
                        'signal':signal,
                        'trigger_date':pd.Timestamp(trigger).strftime('%Y-%m-%d'),
                        'prev_week_high':round(ph,4),
                        'prev_week_low':round(pl,4),
                        'sweep_price':round(sweep,4),
                        'trigger_close':round(close,4),
                        'current_close':round(current_close,4),
                        'volatility_pct':round(vol,2),
                        'distance_pct':round(dist,2)
                    })
            except Exception:
                continue

    latest_trade=max([r['trigger_date'] for r in rows],default=None)
    source=(
        'NSE CM bhavcopy official OHLC; NIFTY 50/500 NSE constituents'
        if market in ('nifty50','nifty500')
        else 'Yahoo Finance daily OHLC; S&P 500 membership from Wikipedia'
    )
    return {
        'generated_at':datetime.now(timezone.utc).isoformat(),
        'data_as_of':now.strftime('%Y-%m-%d %H:%M'),
        'week_start':cur_mon.strftime('%Y-%m-%d'),
        'week_end':(cur_mon+pd.Timedelta(days=4)).strftime('%Y-%m-%d'),
        'source':source,
        'rows':rows,
        'counts':{
            'all':len(rows),
            'high':sum(r['signal']=='HIGH' for r in rows),
            'low':sum(r['signal']=='LOW' for r in rows)
        }
    }


def main():
    os.makedirs('site/data',exist_ok=True)

    nifty50=get_nifty(NIFTY50_URL)
    nifty500=get_nifty(NIFTY_URL)
    markets={
        'nifty50':nifty50,
        'nifty500':nifty500,
        'us500':get_sp500()
    }

    now=pd.Timestamp.now(tz='Asia/Kolkata').tz_localize(None)
    prev_mon,_=week_bounds(now)
    indian_symbols=sorted({sym for items in (nifty50,nifty500) for sym,_ in items})
    nse_cache=load_nse_ohlc(indian_symbols,prev_mon,now)

    for k,v in markets.items():
        data=scan_market(v,k,nse_cache)
        with open(f'site/data/{k}.json','w',encoding='utf-8') as f:
            json.dump(data,f,ensure_ascii=False,indent=2)
        print(k,len(v),'members',len(data['rows']),'fakeouts')


if __name__=='__main__':
    main()
