#!/usr/bin/env python3
"""Generate additional figures for CSI 1000 paper — 图文并茂"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter

OUT = "/mnt/work/qr33/comewealth/figures"
recs = pd.read_csv('/mnt/work/qr33/comewealth/results/csi1000_daily.csv')
trades = pd.read_csv('/mnt/work/qr33/comewealth/results/csi1000_trades.csv')
recs['date'] = pd.to_datetime(recs['date'])

plt.rcParams.update({'font.size': 10, 'figure.dpi': 150, 'axes.grid': True, 'grid.alpha': 0.3})

# === Figure 2: IV Distribution & Percentile Rank ===
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
iv = recs['atm_iv'].dropna() * 100
ax.hist(iv, bins=50, color='#3b82f6', alpha=0.7, edgecolor='white')
ax.axvline(iv.median(), color='red', ls='--', label=f'Median={iv.median():.1f}%')
ax.axvline(iv.mean(), color='orange', ls='--', label=f'Mean={iv.mean():.1f}%')
ax.set(xlabel='ATM IV (%)', ylabel='Frequency', title='A: Distribution of ATM Implied Volatility')
ax.legend(fontsize=8)

ax = axes[1]
rank = recs['iv_rank'].dropna()
ax.hist(rank, bins=50, color='#8b5cf6', alpha=0.7, edgecolor='white')
ax.axvline(80, color='red', ls='--', lw=2, label='Entry Threshold (80th)')
sell_pct = (rank >= 80).mean() * 100
ax.set(xlabel='IV Percentile Rank', ylabel='Frequency', title=f'B: IV Rank Distribution ({sell_pct:.0f}% above threshold)')
ax.legend(fontsize=8)

plt.suptitle('Figure 2: Implied Volatility Characteristics\nAgentic Sciences', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/csi1000_fig2_iv.png', bbox_inches='tight')
plt.close()
print("✅ Figure 2: IV distribution")

# === Figure 3: Monthly P&L Heatmap ===
recs['month'] = recs['date'].dt.to_period('M')
recs['daily_pnl'] = recs['nav'].diff()
monthly = recs.groupby('month')['daily_pnl'].sum() / 1e6  # in millions

fig, ax = plt.subplots(figsize=(10, 4))
colors = ['#dc2626' if x < 0 else '#16a34a' for x in monthly.values]
bars = ax.bar(range(len(monthly)), monthly.values, color=colors, edgecolor='white', width=0.8)
ax.set_xticks(range(0, len(monthly), 3))
ax.set_xticklabels([str(monthly.index[i]) for i in range(0, len(monthly), 3)], rotation=45, fontsize=7)
ax.set(ylabel='P&L (¥M)', title='Monthly P&L Attribution')
ax.axhline(0, color='black', lw=0.5)

# Annotate best and worst
best_idx = monthly.values.argmax()
worst_idx = monthly.values.argmin()
ax.annotate(f'Best: ¥{monthly.values[best_idx]:.1f}M', xy=(best_idx, monthly.values[best_idx]),
            fontsize=7, ha='center', va='bottom', color='#16a34a', fontweight='bold')
ax.annotate(f'Worst: ¥{monthly.values[worst_idx]:.1f}M', xy=(worst_idx, monthly.values[worst_idx]),
            fontsize=7, ha='center', va='top', color='#dc2626', fontweight='bold')

plt.suptitle('Figure 3: Monthly Profit & Loss\nAgentic Sciences', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/csi1000_fig3_monthly.png', bbox_inches='tight')
plt.close()
print("✅ Figure 3: Monthly P&L")

# === Figure 4: Drawdown Analysis ===
nav = recs['nav'].values
peak = np.maximum.accumulate(nav)
dd = (nav - peak) / peak * 100

fig, axes = plt.subplots(2, 1, figsize=(10, 5), gridspec_kw={'height_ratios': [2, 1]})

ax = axes[0]
ax.plot(recs['date'], nav / 1e8, color='#3b82f6', lw=1.5)
ax.fill_between(recs['date'], nav / 1e8, peak / 1e8, alpha=0.15, color='red')
ax.set(ylabel='NAV (¥100M)', title='A: NAV and Peak')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

ax = axes[1]
ax.fill_between(recs['date'], dd, 0, alpha=0.5, color='#dc2626')
ax.set(ylabel='Drawdown (%)', xlabel='Date', title='B: Drawdown from Peak')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

plt.suptitle('Figure 4: Drawdown Analysis\nAgentic Sciences', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/csi1000_fig4_drawdown.png', bbox_inches='tight')
plt.close()
print("✅ Figure 4: Drawdown")

# === Figure 5: Trade P&L Distribution ===
if 'pnl' in trades.columns:
    pnl_col = 'pnl'
elif 'total_pnl' in trades.columns:
    pnl_col = 'total_pnl'
else:
    pnl_col = None
    print(f"Trade columns: {trades.columns.tolist()}")

if pnl_col:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    tpnl = trades[pnl_col] / 1e4  # in ¥10K
    
    ax = axes[0]
    colors_h = ['#dc2626' if x < 0 else '#16a34a' for x in tpnl.values]
    ax.hist(tpnl, bins=40, color='#3b82f6', alpha=0.7, edgecolor='white')
    ax.axvline(0, color='black', lw=1)
    ax.axvline(tpnl.median(), color='red', ls='--', label=f'Median=¥{tpnl.median():.0f}0K')
    ax.set(xlabel='Trade P&L (¥10K)', ylabel='Frequency', title='A: P&L Distribution per Trade')
    ax.legend(fontsize=8)
    
    ax = axes[1]
    cum_pnl = tpnl.cumsum()
    ax.plot(range(len(cum_pnl)), cum_pnl, color='#3b82f6', lw=1.5)
    ax.fill_between(range(len(cum_pnl)), cum_pnl, 0, alpha=0.1, color='#3b82f6')
    ax.set(xlabel='Trade #', ylabel='Cumulative P&L (¥10K)', title='B: Cumulative P&L by Trade Sequence')
    
    plt.suptitle('Figure 5: Trade-Level Analysis\nAgentic Sciences', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT}/csi1000_fig5_trades.png', bbox_inches='tight')
    plt.close()
    print("✅ Figure 5: Trade P&L")

# === Figure 6: Greek Exposures Over Time ===
fig, axes = plt.subplots(2, 2, figsize=(10, 6))

ax = axes[0,0]
ax.plot(recs['date'], recs['gamma']/1e6, color='#dc2626', lw=1)
ax.axhline(-200, color='black', ls='--', lw=0.8, label='Limit (¥200M)')
ax.set(ylabel='¥M', title='A: Portfolio Gamma')
ax.legend(fontsize=7)

ax = axes[0,1]
ax.plot(recs['date'], recs['vega'], color='#059669', lw=1)
ax.axhline(0, color='black', lw=0.5)
ax.set(ylabel='Vega', title='B: Portfolio Vega')

ax = axes[1,0]
ax.plot(recs['date'], recs['delta'], color='#2563eb', lw=1)
ax.axhline(0, color='black', lw=0.5)
ax.set(ylabel='Delta', title='C: Portfolio Delta (after hedging)')

# Panel D: Number of active positions
ax = axes[1,1]
ax.fill_between(recs['date'], recs['n_pos'], 0, alpha=0.5, color='#8b5cf6')
ax.set(ylabel='# Positions', title='D: Active Position Count')

for ax in axes.flat:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
    ax.tick_params(axis='x', rotation=30, labelsize=7)

plt.suptitle('Figure 6: Greek Risk Exposures Over Time\nAgentic Sciences', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/csi1000_fig6_greeks.png', bbox_inches='tight')
plt.close()
print("✅ Figure 6: Greek exposures")

print("\n🎉 All extra figures generated!")
