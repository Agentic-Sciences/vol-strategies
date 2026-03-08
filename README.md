# CSI 1000 Volatility Selling Strategy

Systematic delta-hedged, gamma-controlled short volatility strategy on CFFEX CSI 1000 index options.

**[📄 Read the Paper](https://agentic-sciences.github.io/csi1000-vol-selling/assets/csi1000_vol_selling.pdf)** | **[🌐 Live Site](https://agentic-sciences.github.io/csi1000-vol-selling/)**

## Key Results

| Metric | Value |
|---|---|
| Period | Aug 2022 – Mar 2026 |
| Total Return | 71.9% |
| Annualized | 16.5% |
| Sharpe Ratio | 0.94 |
| Max Drawdown | -12.3% |
| Win Rate | 63.3% |

## Repository Structure

```
├── index.html              # Research website
├── assets/
│   ├── csi1000_vol_selling.pdf   # Full paper (76 pages)
│   ├── figures/            # All research figures
│   └── data/               # Daily strategy results (CSV)
└── code/
    ├── csi1000_vol_selling.py    # Strategy backtest engine
    ├── csi1000_download.py       # Option data downloader (AKShare)
    ├── gen_csi1000_paper.py      # PDF paper generator
    └── csi1000_extra_figs.py     # Figure generator
```

## Data Sources

All data is publicly available via [AKShare](https://github.com/akfamily/akshare) (free, open-source).

---

*Agentic Sciences — Research. Content. Capital.*
