# option-pricing-engine

# Option Pricing Engine

A full-stack quantitative finance project that prices SPY options using three industry-standard models — **Black-Scholes**, **CRR Binomial Tree**, and **Monte Carlo simulation** — against live market data from Yahoo Finance.

Built as a portfolio project for breaking into quantitative finance. The project covers the complete pipeline from raw option chain extraction through model implementation, error analysis, and interactive deployment.

🔗 **[Live Demo — Streamlit Dashboard](https://option-pricing-engine-main.streamlit.app/)**

---

## Overview

Most people learn option pricing models from a textbook. This project builds them from scratch, stress-tests them against live SPY market data, and compares their outputs side by side.

The core insight: when you back out implied volatility from market prices and feed it into your model, errors shrink to near zero — not because the model is good, but because you've already used the market price to calibrate it. Understanding that distinction is what this project is really about.

---

## Models Implemented

### Black-Scholes
Closed-form European option pricer under the assumption of constant volatility and log-normally distributed returns. Implemented using the risk-neutral pricing framework with `scipy.stats.norm` for the cumulative normal distribution. Serves as the baseline for all comparisons.

### Implied Volatility Solver
Inverts the Black-Scholes formula numerically to recover the market-implied volatility from observed option prices. Uses **Brent's method** (`scipy.optimize.brentq`) with bracket expansion for robustness on deep ITM/OTM contracts. Returns `NaN` gracefully when no root exists.

### CRR Binomial Tree
Cox-Ross-Rubinstein binomial tree with backward induction for European options. Up/down factors derived from volatility to guarantee convergence to the Black-Scholes price as steps increase. Vectorised terminal payoff computation with O(N) space complexity via in-place backward induction.

### Monte Carlo (GBM)
Simulates asset price paths under Geometric Brownian Motion and discounts expected payoffs. Implements **antithetic variates** for variance reduction — pairing each random draw `z` with `-z` halves variance without increasing the number of model evaluations. Returns both price and standard error. A convergence study quantifies the 1/√N error decay rate.

---

## Project Structure

```
option_pricing/
├── app.py                        # Streamlit dashboard
├── utils.py                      # Shared pricing functions (imported by all notebooks)
├── requirements.txt
├── .streamlit/
│   └── config.toml               # Theme and server config
├── 01_data_extraction.ipynb      # yfinance data pipeline → saves opts_clean.parquet
├── 02_black_scholes_iv.ipynb     # BS pricer + IV solver → saves opts_with_bs.parquet
├── 03_binomial_crr.ipynb         # CRR binomial tree → saves opts_with_binomial.parquet
├── 04_monte_carlo.ipynb          # MC simulation + convergence study → opts_final.parquet
├── data/                         # Parquet files (generated at runtime, not tracked)
└── plots/                        # HTML chart outputs (generated at runtime, not tracked)
```

### Why this structure?
Each notebook is fully independent — it reads a parquet file from the previous stage and saves its own. This means you never have to re-run the yfinance data fetch to test a model change, and the kernel restart problem disappears. All shared functions (`bs_price`, `binomial_price`, `mc_price`, `bs_implied_vol_single`, `safe_write_html`) live once in `utils.py` and are imported everywhere.

---

## Streamlit Dashboard

The interactive dashboard allows you to:

- **Single Option Pricer** — adjust spot, strike, vol, and maturity via sliders and watch all three model prices update instantly
- **Live Market Data** — fetch live SPY option chains (or any ticker) directly from Yahoo Finance
- **Expiry Selector** — switch between expiry dates and reprice the full chain
- **Three-Model Comparison Table** — BS, CRR, and MC prices alongside market mid, with MAE per model
- **Error vs Strike** — scatter plot of pricing error across the strike range, switchable between models
- **MC Convergence Study** — plots |MC − BS| and standard error against path count on the nearest ATM contract

---

## Installation & Running Locally

**1. Clone the repo and create a virtual environment**
```bash
git clone https://github.com/ParthaSarathiChakraborty/option-pricing-engine.git 
cd option-pricing-engine
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Create the data and plots directories**
```bash
mkdir data plots
```

**4. Run the notebooks in order** (or use the shell script)
```bash
bash run_all.sh
# or manually in Jupyter — Kernel → Restart & Run All, one notebook at a time
```

**5. Launch the Streamlit app**
```bash
streamlit run app.py
```

---

## Running All Notebooks at Once

A `run_all.sh` shell script executes all four notebooks sequentially using `nbconvert`:

```bash
#!/bin/bash
set -e
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 01_data_extraction.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 02_black_scholes_iv.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 03_binomial_crr.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 04_monte_carlo.ipynb
```

Monte Carlo gets a 600s timeout since it is the heaviest notebook.

---

## Dependencies

| Package | Purpose |
|---|---|
| `yfinance` | Live option chain and spot price data |
| `pandas` | Data cleaning, filtering, and manipulation |
| `numpy` | Vectorised numerical computation |
| `scipy` | `norm.cdf` for BS formula, `brentq` for IV solver |
| `plotly` | Interactive visualisations |
| `streamlit` | Dashboard deployment |
| `pyarrow` | Parquet file I/O between notebooks |

---

## Key Design Decisions

**Mid price over last price** — bid/ask midpoint is a cleaner proxy for fair value than the last traded price, which may be stale. Falls back to `lastPrice` when bid/ask are unavailable or zero.

**No liquidity filter** — yfinance returns zero `openInterest` and `volume` on many actively traded contracts. A liquidity filter that relies on these fields silently removes the entire dataset on certain API calls.

**Brent's method over Newton-Raphson** — guaranteed convergence without requiring the Vega derivative, and handles edge cases (deep ITM/OTM, near-expiry contracts) more robustly.

**Antithetic variates** — pairing `z` with `-z` in Monte Carlo reduces variance by exploiting negative correlation between paired paths. Same accuracy as doubling the path count at half the computational cost.

**Parquet over CSV** — preserves dtypes (datetime columns especially), is faster to read/write, and compresses better for large option chain DataFrames.

**`utils.py` as a single source of truth** — all four notebooks and the Streamlit app import from the same module. A bug fix in `bs_price` propagates everywhere instantly.

---

## Known Limitations

- **Constant volatility assumption** — Black-Scholes assumes σ is constant, which the IV smile directly contradicts. A stochastic volatility model (e.g. Heston) would address this.
- **European options only** — the binomial tree can be extended to American options by adding an early exercise check during backward induction.
- **Hardcoded risk-free rate** — currently defaults to 5%. Pulling the 3-month T-bill yield from FRED would make this live and accurate.
- **yfinance rate limiting** — Yahoo Finance throttles requests from cloud server IPs. The app handles this with retry logic and exponential backoff, but a commercial data feed (Bloomberg, Refinitiv) would be more reliable in production.

---

## Mathematical Background

The project touches the following concepts from quantitative finance and stochastic calculus:

- Risk-neutral pricing and no-arbitrage conditions
- Geometric Brownian Motion and the log-normal stock price model
- Itô's lemma applied to the Black-Scholes PDE
- The connection between PDEs and expectations (Feynman-Kac)
- Implied volatility and the volatility smile
- Binomial tree convergence to continuous-time models
- Monte Carlo variance reduction techniques

---

## What's Next

- Greeks computation (Delta, Gamma, Vega, Theta) via finite differences
- American option pricing via early exercise check in the binomial tree
- Heston stochastic volatility model to address the IV smile limitation
- Live risk-free rate fetch from FRED to replace the hardcoded placeholder

---

## Disclaimer

This project is for educational purposes only. Nothing in this repository constitutes financial advice. Option pricing models make simplifying assumptions that do not hold in real markets.

---

*Built by Partha Sarathi Chakraborty — aspiring quantitative analyst*
*[LinkedIn](https://www.linkedin.com/in/parthasarathicharavarty/) · [Live Demo](https://option-pricing-engine-main.streamlit.app/)*
