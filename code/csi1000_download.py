#!/usr/bin/env python3
"""Download CSI 1000 option data with retry and incremental caching."""
import akshare as ak
import pandas as pd
import time
from pathlib import Path
from datetime import datetime
from calendar import monthcalendar

OUT = Path('/mnt/work/qr33/comewealth/cache')
OUT.mkdir(exist_ok=True)
CACHE = OUT / 'csi1000_options.parquet'

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def expiry_date(ms):
    y, m = 2000 + int(ms[2:4]), int(ms[4:6])
    fridays = [w[4] for w in monthcalendar(y, m) if w[4] != 0]
    return datetime(y, m, fridays[2])

# Load index for ATM estimation
log("Loading index...")
idx = ak.stock_zh_index_daily(symbol="sh000852")
idx['date'] = pd.to_datetime(idx['date'])
idx = idx[idx['date'] >= '2022-07-01'].set_index('date').sort_index()

# Load existing cache
existing = set()
if CACHE.exists():
    old = pd.read_parquet(CACHE)
    existing = set(old['symbol'].unique())
    log(f"Existing cache: {len(old):,} obs, {len(existing)} series")
    all_data = [old]
else:
    all_data = []

# Generate months
months = []
dt = datetime(2022, 8, 1)
while dt <= datetime(2026, 6, 1):
    months.append(f"mo{dt.strftime('%y%m')}")
    dt = datetime(dt.year + (dt.month // 12), (dt.month % 12) + 1, 1)
months = sorted(set(months))

downloaded = 0
for month in months:
    try: exp = expiry_date(month)
    except: continue
    mask = idx.index <= exp
    if mask.sum() == 0: continue
    level = idx.loc[mask, 'close'].iloc[-1]
    center = int(round(level / 100) * 100)
    mc = 0
    for strike in range(center - 500, center + 600, 100):
        for cp in ['C', 'P']:
            sym = f"{month}{cp}{strike}"
            if sym in existing: continue
            for attempt in range(3):
                try:
                    df = ak.option_cffex_zz1000_daily_sina(symbol=sym)
                    if len(df) > 0:
                        df['symbol'] = sym
                        df['month'] = month
                        df['strike'] = strike
                        df['cp'] = cp
                        df['expiry'] = exp
                        all_data.append(df)
                        downloaded += 1; mc += 1
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(2)
                    else:
                        pass
    if mc: log(f"  {month} (ATM~{center}): {mc} new series")

log(f"Downloaded {downloaded} new series")

if downloaded > 0 or len(all_data) > 0:
    result = pd.concat(all_data, ignore_index=True)
    result['date'] = pd.to_datetime(result['date'])
    result = result.drop_duplicates(subset=['date', 'symbol']).sort_values(['date', 'symbol'])
    result.to_parquet(CACHE)
    log(f"Saved: {len(result):,} obs, {result['symbol'].nunique()} series → {CACHE}")
