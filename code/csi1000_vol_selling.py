#!/usr/bin/env python3
"""
CSI 1000 Index Option Volatility Selling Strategy
===================================================
Sell ATM straddle when IV > 80th percentile, delta-hedge with futures.
Risk: per-trade gamma ≤ ¥2M, portfolio gamma ≤ ¥200M.
Data: AKShare (Sina) — CSI 1000 options since 2022-07.

Agentic Sciences | March 2026
"""

import akshare as ak
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, time, warnings
from pathlib import Path
from datetime import datetime
from calendar import monthcalendar
warnings.filterwarnings('ignore')

OUT = Path('/mnt/work/qr33/comewealth')
CACHE = OUT / 'cache/csi1000_options.parquet'
t0 = time.time()
def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ===== Parameters =====
IV_PCTILE = 80
MAX_TRADE_GAMMA = 2_000_000
MAX_PORT_GAMMA  = 200_000_000
MULTIPLIER = 100
INITIAL_CAPITAL = 100_000_000
R_FREE = 0.025

# ===== Black-Scholes =====
def bs_price(S, K, T, r, sigma, cp):
    if T <= 0 or sigma <= 0: return max(0, (S-K) if cp=='C' else (K-S))
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2) if cp=='C' \
        else K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def bs_greeks(S, K, T, r, sigma, cp):
    if T <= 0 or sigma <= 0: return 0, 0, 0, 0
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    delta = norm.cdf(d1) if cp=='C' else norm.cdf(d1)-1
    theta = -S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2) if cp=='C' \
        else -S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2)
    return delta, gamma, vega, theta

def implied_vol(price, S, K, T, r, cp):
    if T <= 0 or price <= 0: return np.nan
    intrinsic = max(0, (S-K) if cp=='C' else (K-S))
    if price <= intrinsic: return np.nan
    try: return brentq(lambda s: bs_price(S, K, T, r, s, cp) - price, 0.01, 5.0, xtol=1e-6)
    except: return np.nan

def expiry_date(month_str):
    y, m = 2000 + int(month_str[2:4]), int(month_str[4:6])
    fridays = [w[4] for w in monthcalendar(y, m) if w[4] != 0]
    return datetime(y, m, fridays[2])

# ===== Step 1: CSI 1000 Index =====
log("Loading CSI 1000 index...")
idx = ak.stock_zh_index_daily(symbol="sh000852")
idx['date'] = pd.to_datetime(idx['date'])
idx = idx[idx['date'] >= '2022-07-01'].set_index('date').sort_index()
log(f"Index: {len(idx)} days ({idx.index[0].date()} → {idx.index[-1].date()})")

# ===== Step 2: Download or load cached option data =====
if CACHE.exists():
    log(f"Loading cached data...")
    raw = pd.read_parquet(CACHE)
    downloaded = raw['symbol'].nunique()
    log(f"Cached: {len(raw):,} obs, {downloaded} series")
else:
    log("Downloading option contracts from Sina...")
    months = []
    dt = datetime(2022, 8, 1)
    while dt <= datetime(2026, 6, 1):
        months.append(f"mo{dt.strftime('%y%m')}")
        dt = datetime(dt.year + (dt.month // 12), (dt.month % 12) + 1, 1)
    months = sorted(set(months))
    log(f"Contract months: {len(months)}")

    all_data = []
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
                try:
                    df = ak.option_cffex_zz1000_daily_sina(symbol=f"{month}{cp}{strike}")
                    if len(df) > 0:
                        df['symbol'] = f"{month}{cp}{strike}"
                        df['month'] = month
                        df['strike'] = strike
                        df['cp'] = cp
                        df['expiry'] = exp
                        all_data.append(df)
                        downloaded += 1; mc += 1
                except: pass
        if mc: log(f"  {month} (ATM~{center}): {mc} series")

    log(f"Total: {downloaded} series")
    if downloaded == 0:
        log("ERROR: No data"); exit(1)

    raw = pd.concat(all_data, ignore_index=True)
    raw['date'] = pd.to_datetime(raw['date'])
    raw = raw.sort_values(['date', 'symbol']).reset_index(drop=True)
    raw.to_parquet(CACHE)
    log(f"Cached to {CACHE}")

log(f"Raw observations: {len(raw):,}")

# ===== Step 3: Compute IV & Greeks =====
log("Computing IV and Greeks...")
opts = raw.merge(idx[['close']].rename(columns={'close': 'S'}),
                 left_on='date', right_index=True, how='left').dropna(subset=['S'])
opts['ttm'] = (pd.to_datetime(opts['expiry']) - pd.to_datetime(opts['date'])).dt.days / 365.25
opts = opts[opts['ttm'] > 0.01].copy()
for c in ['close', 'S', 'strike', 'ttm']:
    opts[c] = pd.to_numeric(opts[c], errors='coerce')
opts = opts.dropna(subset=['close', 'S', 'strike', 'ttm'])

opts['iv'] = [implied_vol(float(r['close']), float(r['S']), float(r['strike']),
              float(r['ttm']), R_FREE, r['cp']) for _, r in opts.iterrows()]
opts = opts.dropna(subset=['iv'])
opts = opts[(opts['iv'] > 0.05) & (opts['iv'] < 3.0)].copy()

greeks = [bs_greeks(float(r['S']), float(r['strike']), float(r['ttm']),
          R_FREE, float(r['iv']), r['cp']) for _, r in opts.iterrows()]
opts[['delta','gamma','vega','theta']] = pd.DataFrame(greeks, index=opts.index)
log(f"After IV: {len(opts):,} obs")

# ===== Step 4: ATM IV Series =====
log("Building ATM IV time series...")
opts['moneyness'] = abs(opts['strike'] / opts['S'] - 1)
daily_iv = opts[opts['moneyness'] < 0.03].groupby('date')['iv'].median()
iv_rank = daily_iv.rolling(252, min_periods=20).rank(pct=True) * 100
signals = pd.DataFrame({'atm_iv': daily_iv, 'iv_rank': iv_rank}).dropna()
signals = signals.join(idx[['close']].rename(columns={'close': 'S'})).dropna()
log(f"Signal days: {len(signals)}")

# ===== Step 5: Strategy =====
log("Running strategy...")
capital = INITIAL_CAPITAL
positions = []
records = []
trade_log = []
prev_S = None

for date, sig in signals.iterrows():
    S = sig['S']

    # Update hedge P&L
    if prev_S and prev_S > 0:
        dr = S / prev_S - 1
        for p in positions:
            p['hedge_pnl'] += (-p['cur_delta']) * MULTIPLIER * p['n'] * S * dr
            T_now = (p['expiry'] - date).days / 365.25
            if T_now > 0.01:
                d, g, v, _ = bs_greeks(S, p['strike'], T_now, R_FREE, p['iv_entry'], p['cp'])
                p['cur_delta'], p['cur_gamma'], p['cur_vega'] = d, g, v

    # Expire
    active, expired = [], []
    for p in positions:
        ((expired if (p['expiry'] - date).days <= 0 else active)).append(p)
    for p in expired:
        payoff = max(0, S - p['strike']) if p['cp']=='C' else max(0, p['strike'] - S)
        pnl = p['premium_per'] * p['n'] * MULTIPLIER - payoff * p['n'] * MULTIPLIER + p['hedge_pnl']
        capital += pnl
        trade_log.append({
            'entry': p['entry'], 'exit': date, 'cp': p['cp'], 'strike': p['strike'],
            'iv': p['iv_entry'], 'iv_rank': p['iv_rank'], 'n': p['n'],
            'premium': p['premium_per']*p['n']*MULTIPLIER, 'payoff': payoff*p['n']*MULTIPLIER,
            'hedge_pnl': p['hedge_pnl'], 'total_pnl': pnl,
        })
    positions = active

    # Portfolio greeks
    pg = sum(p['cur_gamma'] * MULTIPLIER * p['n'] * S**2 / 100 for p in positions)
    pd_ = sum(p['cur_delta'] * MULTIPLIER * p['n'] for p in positions)
    pv = sum(p['cur_vega'] * MULTIPLIER * p['n'] for p in positions)

    action = 'hold'
    n_c = n_p = 0
    prem = 0

    if sig['iv_rank'] >= IV_PCTILE and pg < MAX_PORT_GAMMA:
        day_opts = opts[(opts['date'] == date) & (opts['moneyness'] < 0.02)]
        if len(day_opts) > 0:
            near = day_opts.sort_values('ttm')
            near = near[near['ttm'] < near['ttm'].iloc[0] + 0.05]
            for cp in ['C', 'P']:
                side = near[near['cp'] == cp].sort_values('moneyness')
                if len(side) == 0: continue
                o = side.iloc[0]
                gpc = o['gamma'] * MULTIPLIER * S**2 / 100
                if gpc <= 0: continue
                n = min(int(MAX_TRADE_GAMMA / gpc), int((MAX_PORT_GAMMA - pg) / gpc), 50)
                if n <= 0: continue
                positions.append({
                    'entry': date, 'cp': cp, 'strike': o['strike'],
                    'expiry': o['expiry'], 'iv_entry': o['iv'], 'iv_rank': sig['iv_rank'],
                    'premium_per': o['close'], 'n': n,
                    'cur_delta': o['delta'], 'cur_gamma': o['gamma'], 'cur_vega': o['vega'],
                    'hedge_pnl': 0,
                })
                pg += gpc * n
                prem += o['close'] * n * MULTIPLIER
                if cp == 'C': n_c = n
                else: n_p = n
            if n_c + n_p > 0: action = 'sell'

    records.append({
        'date': date, 'index': S, 'atm_iv': sig['atm_iv'], 'iv_rank': sig['iv_rank'],
        'action': action, 'sold_C': n_c, 'sold_P': n_p, 'premium': prem,
        'n_pos': len(positions), 'gamma': -pg, 'delta': pd_, 'vega': -pv,
        'nav': capital,
    })
    prev_S = S

# Force-expire remaining
for p in positions:
    S = signals.iloc[-1]['S']
    payoff = max(0, S - p['strike']) if p['cp']=='C' else max(0, p['strike'] - S)
    pnl = p['premium_per']*p['n']*MULTIPLIER - payoff*p['n']*MULTIPLIER + p['hedge_pnl']
    capital += pnl
    trade_log.append({
        'entry': p['entry'], 'exit': signals.index[-1], 'cp': p['cp'],
        'strike': p['strike'], 'iv': p['iv_entry'], 'iv_rank': p['iv_rank'], 'n': p['n'],
        'premium': p['premium_per']*p['n']*MULTIPLIER, 'payoff': payoff*p['n']*MULTIPLIER,
        'hedge_pnl': p['hedge_pnl'], 'total_pnl': pnl,
    })

recs = pd.DataFrame(records)
tlog = pd.DataFrame(trade_log)

# ===== Results =====
ret = capital / INITIAL_CAPITAL - 1
ny = (recs['date'].iloc[-1] - recs['date'].iloc[0]).days / 365.25
ann = (1+ret)**(1/ny)-1 if ny > 0 else 0
recs['r'] = recs['nav'].pct_change().fillna(0)
sharpe = recs['r'].mean() / recs['r'].std() * np.sqrt(252) if recs['r'].std() > 0 else 0
dd = (recs['nav'] / recs['nav'].cummax() - 1).min()
wr = (tlog['total_pnl'] > 0).mean() if len(tlog) > 0 else 0

log(f"\n{'='*50}")
log(f"CSI 1000 Vol Selling Results")
log(f"{'='*50}")
log(f"Period: {recs['date'].iloc[0].date()} → {recs['date'].iloc[-1].date()}")
log(f"NAV: ¥{capital:,.0f}")
log(f"Return: {ret:.2%} | Ann: {ann:.2%}")
log(f"Sharpe: {sharpe:.2f} | DD: {dd:.2%}")
log(f"Win: {wr:.1%} | Trades: {len(tlog)} | Sell days: {(recs['action']=='sell').sum()}")

stats = {
    'instrument': 'CSI 1000 Index Options (MO)', 'exchange': 'CFFEX',
    'period': f"{recs['date'].iloc[0].date()} to {recs['date'].iloc[-1].date()}",
    'initial_capital': INITIAL_CAPITAL, 'final_nav': float(capital),
    'total_return': float(ret), 'ann_return': float(ann),
    'sharpe': float(sharpe), 'max_dd': float(dd), 'win_rate': float(wr),
    'total_trades': len(tlog), 'option_obs': len(opts),
    'risk': {'max_trade_gamma': MAX_TRADE_GAMMA, 'max_port_gamma': MAX_PORT_GAMMA},
}

# ===== Figure =====
log("Generating figure...")
fig, axes = plt.subplots(3, 2, figsize=(16, 14), facecolor='white')

ax = axes[0,0]
ax.plot(recs['date'], recs['nav']/1e6, c='#2563eb', lw=2)
ax.axhline(INITIAL_CAPITAL/1e6, c='k', ls='--', lw=1, alpha=.5)
ax.set(title='A: Strategy NAV (¥M)', ylabel='¥M'); ax.grid(alpha=.3)

ax = axes[0,1]
ax.plot(recs['date'], recs['index'], c='#6b7280', lw=1, label='CSI 1000')
ax2 = ax.twinx()
ax2.fill_between(recs['date'], recs['atm_iv']*100, alpha=.3, color='#f59e0b')
ax.set(title='B: CSI 1000 Index & ATM IV'); ax2.set_ylabel('IV (%)'); ax.grid(alpha=.3)

ax = axes[1,0]
ax.plot(recs['date'], recs['gamma']/1e6, c='#dc2626', lw=1.5)
ax.axhline(-MAX_PORT_GAMMA/1e6, c='k', ls='--', lw=1, label=f'Limit')
ax.set(title='C: Portfolio Gamma (¥M)', ylabel='¥M'); ax.legend(); ax.grid(alpha=.3)

ax = axes[1,1]
ax.plot(recs['date'], recs['delta'], c='#7c3aed', lw=1.5)
ax.set(title='D: Portfolio Delta', ylabel='Delta'); ax.grid(alpha=.3)

ax = axes[2,0]
ax.fill_between(recs['date'], recs['iv_rank'], alpha=.4, color='#f59e0b')
ax.axhline(IV_PCTILE, c='#dc2626', ls='--', lw=1.5)
sells = recs[recs['action']=='sell']
ax.scatter(sells['date'], sells['iv_rank'], c='red', s=20, zorder=5)
ax.set(title='E: IV Rank & Signals', ylabel='Percentile'); ax.grid(alpha=.3)

ax = axes[2,1]
ax.plot(recs['date'], recs['vega'], c='#059669', lw=1.5)
ax.set(title='F: Portfolio Vega', ylabel='Vega'); ax.grid(alpha=.3)

plt.suptitle('CSI 1000 Index Option Vol Selling (IV>80th, Gamma-Controlled, Δ-Hedged)\nAgentic Sciences',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT/'figures/csi1000_vol_selling.png', dpi=150, bbox_inches='tight')

recs.to_csv(OUT/'results/csi1000_daily.csv', index=False)
if len(tlog) > 0: tlog.to_csv(OUT/'results/csi1000_trades.csv', index=False)
json.dump(stats, open(OUT/'results/csi1000_summary.json','w'), indent=2)

log(f"Done in {time.time()-t0:.0f}s")
