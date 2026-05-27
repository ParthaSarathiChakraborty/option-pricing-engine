"""
utils.py — Option Pricing Shared Utilities
===========================================
Central module for all pricing functions and helpers.
Import from any notebook with: from utils import bs_price, bs_implied_vol_single, ...

Functions
---------
safe_write_html       : Robust Plotly HTML saver with fallback dirs
bs_price              : Black-Scholes pricer (numpy-based, European only)
bs_implied_vol_single : Implied vol solver via Brent's method
binomial_price        : CRR Binomial Tree — supports European AND American options
mc_price              : Monte Carlo GBM pricer with antithetic variates (European)
early_exercise_premium: Convenience function — American minus European binomial price
"""

import os
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


# ─────────────────────────────────────────────
# I/O HELPER
# ─────────────────────────────────────────────

def safe_write_html(fig, filename, preferred_dir="plots"):
    """
    Save a Plotly figure as HTML, trying preferred_dir first,
    then ./plots, then the current working directory.
    """
    candidates = [preferred_dir, "plots", os.getcwd()]
    last_exc   = None
    for d in candidates:
        try:
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            out_path = os.path.join(d if d else "", filename)
            fig.write_html(out_path)
            print(f"Saved: {out_path}")
            return out_path
        except Exception as e:
            last_exc = e
            continue
    raise RuntimeError(f"Failed to save {filename}. Last error: {last_exc}")


# ─────────────────────────────────────────────
# BLACK-SCHOLES  (European only)
# ─────────────────────────────────────────────

def bs_price(spot, K, T, r, sigma, option_type="call"):
    """
    Black-Scholes price for a European option.

    Uses numpy throughout so it is compatible with vectorised
    apply() operations without modification.

    Parameters
    ----------
    spot, K, T, r, sigma : floats
    option_type          : 'call' or 'put'

    Returns
    -------
    float : option price
    """
    if T <= 0 or sigma <= 0:
        intrinsic = max(spot - K, 0.0) if option_type == "call" else max(K - spot, 0.0)
        return float(intrinsic)

    d1 = (np.log(spot / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = spot * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - spot * norm.cdf(-d1)

    return float(price)


# ─────────────────────────────────────────────
# IMPLIED VOLATILITY SOLVER
# ─────────────────────────────────────────────

def bs_implied_vol_single(market_price, spot, K, T, r,
                           option_type="call", tol=1e-6, maxiter=100):
    """
    Solve for implied vol using Brent's method.

    Returns np.nan if the solver fails to bracket a root
    (e.g. deep ITM options with no time value).
    """
    intrinsic = max(0.0, (spot - K) if option_type == "call" else (K - spot))
    if market_price <= intrinsic + 1e-12:
        return 1e-12

    def f(sigma):
        return bs_price(spot, K, T, r, sigma, option_type) - market_price

    low, high = 1e-6, 5.0
    try:
        fl, fh = f(low), f(high)
        if fl * fh > 0:
            high = 10.0
            fh   = f(high)
            if fl * fh > 0:
                return np.nan
        return float(brentq(f, low, high, xtol=tol, maxiter=maxiter))
    except Exception:
        return np.nan


# ─────────────────────────────────────────────
# BINOMIAL TREE (CRR)  — European & American
# ─────────────────────────────────────────────

def binomial_price(spot, K, T, r, sigma,
                   steps=100, option_type="call", option_style="european"):
    """
    Cox-Ross-Rubinstein Binomial Tree pricer.

    Supports both European and American exercise styles.

    Parameters
    ----------
    spot, K, T, r, sigma : floats
    steps                : int — number of time steps (default 100)
    option_type          : 'call' or 'put'
    option_style         : 'european' or 'american'

        European: holder can only exercise at expiry.
        American: holder can exercise at any node during backward
                  induction. At each node we take:
                      max(continuation_value, exercise_value)
                  This is the only difference from European pricing.

    Returns
    -------
    float : option price, or np.nan on invalid inputs

    Notes
    -----
    Early exercise of an American call is never optimal when there
    are no dividends — the early exercise premium will be zero for
    calls on non-dividend paying assets (call-put parity argument).
    The premium is meaningful for American puts, especially deep ITM
    puts close to expiry where interest on the strike exceeds the
    remaining time value.
    """
    if T <= 0 or sigma <= 0 or spot <= 0 or K <= 0:
        return np.nan

    dt = T / steps
    u  = np.exp(sigma * np.sqrt(dt))
    d  = 1.0 / u
    p  = (np.exp(r * dt) - d) / (u - d)

    if not (0.0 < p < 1.0):
        return np.nan

    # ── Terminal asset prices (vectorised) ──────────────────────────
    j      = np.arange(steps + 1)
    prices = spot * (u ** j) * (d ** (steps - j))

    # ── Terminal payoffs ─────────────────────────────────────────────
    if option_type == "call":
        values = np.maximum(prices - K, 0.0)
    else:
        values = np.maximum(K - prices, 0.0)

    # ── Backward induction ───────────────────────────────────────────
    discount = np.exp(-r * dt)

    for step in range(steps):
        # Asset prices at this node level (steps - step - 1 remaining)
        node_prices = spot * (u ** np.arange(steps - step)) * (d ** np.arange(steps - step - 1, -1, -1))

        # Continuation value
        continuation = discount * (p * values[1:] + (1.0 - p) * values[:-1])

        if option_style == "american":
            # Exercise value at each node
            if option_type == "call":
                exercise = np.maximum(node_prices - K, 0.0)
            else:
                exercise = np.maximum(K - node_prices, 0.0)

            # American: take the best of hold vs exercise at every node
            values = np.maximum(continuation, exercise)
        else:
            # European: no early exercise — continuation value only
            values = continuation

    return float(values[0])


# ─────────────────────────────────────────────
# EARLY EXERCISE PREMIUM
# ─────────────────────────────────────────────

def early_exercise_premium(spot, K, T, r, sigma,
                            steps=100, option_type="put"):
    """
    Compute the early exercise premium: American price − European price.

    The premium is always >= 0 by no-arbitrage (an American option
    is always worth at least as much as an equivalent European option).

    For calls on non-dividend paying assets, this will be ~0 since
    early exercise of calls is never optimal.

    For puts, the premium is largest for:
    - Deep ITM options (high intrinsic value, low time value)
    - High interest rates (large opportunity cost of holding the put)
    - Short maturities (less time value to sacrifice)

    Parameters
    ----------
    spot, K, T, r, sigma : floats
    steps                : int
    option_type          : 'call' or 'put'

    Returns
    -------
    (american_price, european_price, premium) : tuple of floats
    """
    american = binomial_price(spot, K, T, r, sigma,
                              steps=steps, option_type=option_type,
                              option_style="american")
    european = binomial_price(spot, K, T, r, sigma,
                              steps=steps, option_type=option_type,
                              option_style="european")

    if np.isnan(american) or np.isnan(european):
        return np.nan, np.nan, np.nan

    premium = float(american - european)
    return float(american), float(european), max(premium, 0.0)


# ─────────────────────────────────────────────
# MONTE CARLO (GBM)  — European only
# ─────────────────────────────────────────────

def mc_price(spot, K, T, r, sigma,
             n_paths=100_000, option_type="call",
             seed=None, antithetic=True):
    """
    Monte Carlo pricer for European options under GBM.

    Note: MC does not support American options in this implementation.
    American options require simulation of the full path AND an optimal
    stopping rule at each time step (Longstaff-Schwartz / LSM), which
    is a significant extension beyond the current scope.

    Parameters
    ----------
    n_paths    : number of simulated paths
    seed       : int or None — pass None for non-deterministic runs,
                 pass an int for reproducible single-option tests
    antithetic : if True, pairs each z with -z to reduce variance
                 without increasing the number of model evaluations

    Returns
    -------
    (price, std_err) : tuple of floats
                       Returns (np.nan, np.nan) on invalid inputs
                       so unpacking is always safe.
    """
    if T <= 0 or sigma <= 0 or spot <= 0 or K <= 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    if antithetic:
        half = n_paths // 2
        z    = rng.standard_normal(half)
        z    = np.concatenate([z, -z])
        if len(z) < n_paths:
            z = np.concatenate([z, rng.standard_normal(1)])
    else:
        z = rng.standard_normal(n_paths)

    ST = spot * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)

    payoffs  = np.maximum(ST - K, 0.0) if option_type == "call" else np.maximum(K - ST, 0.0)
    discount = np.exp(-r * T)

    price   = discount * np.mean(payoffs)
    std_err = discount * np.std(payoffs) / np.sqrt(len(payoffs))

    return float(price), float(std_err)
