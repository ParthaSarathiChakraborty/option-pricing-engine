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
bs_greeks             : All five first-order Greeks from Black-Scholes analytically
compute_vega          : Standalone Vega per unit vol move (used in surface notebook)
fit_vol_surface       : Fit a 2D interpolated vol surface from scattered IV data
query_surface_iv      : Query the fitted surface at arbitrary (log_moneyness, T) points
"""

import os
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import griddata


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

    Returns
    -------
    float : option price, or np.nan on invalid inputs
    """
    if T <= 0 or sigma <= 0 or spot <= 0 or K <= 0:
        return np.nan

    dt = T / steps
    u  = np.exp(sigma * np.sqrt(dt))
    d  = 1.0 / u
    p  = (np.exp(r * dt) - d) / (u - d)

    if not (0.0 < p < 1.0):
        return np.nan

    # Terminal asset prices (vectorised)
    j      = np.arange(steps + 1)
    prices = spot * (u ** j) * (d ** (steps - j))

    # Terminal payoffs
    if option_type == "call":
        values = np.maximum(prices - K, 0.0)
    else:
        values = np.maximum(K - prices, 0.0)

    # Backward induction
    discount = np.exp(-r * dt)

    for step in range(steps):
        node_prices  = spot * (u ** np.arange(steps - step)) * \
                       (d ** np.arange(steps - step - 1, -1, -1))
        continuation = discount * (p * values[1:] + (1.0 - p) * values[:-1])

        if option_style == "american":
            if option_type == "call":
                exercise = np.maximum(node_prices - K, 0.0)
            else:
                exercise = np.maximum(K - node_prices, 0.0)
            values = np.maximum(continuation, exercise)
        else:
            values = continuation

    return float(values[0])


# ─────────────────────────────────────────────
# EARLY EXERCISE PREMIUM
# ─────────────────────────────────────────────

def early_exercise_premium(spot, K, T, r, sigma,
                            steps=100, option_type="put"):
    """
    Compute the early exercise premium: American price - European price.

    The premium is always >= 0 by no-arbitrage.
    For calls on non-dividend paying assets this will be ~0.
    For puts, the premium is largest for deep ITM, high-r, short-T contracts.

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

    Parameters
    ----------
    n_paths    : number of simulated paths
    seed       : int or None
    antithetic : if True, pairs each z with -z to reduce variance

    Returns
    -------
    (price, std_err) : tuple — always a tuple, (nan, nan) on bad inputs
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

    ST       = spot * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)
    payoffs  = np.maximum(ST - K, 0.0) if option_type == "call" else np.maximum(K - ST, 0.0)
    discount = np.exp(-r * T)

    price   = discount * np.mean(payoffs)
    std_err = discount * np.std(payoffs) / np.sqrt(len(payoffs))

    return float(price), float(std_err)


# ─────────────────────────────────────────────
# BLACK-SCHOLES GREEKS  (analytic)
# ─────────────────────────────────────────────

def bs_greeks(spot, K, T, r, sigma, option_type="call"):
    """
    Compute all five first-order Black-Scholes Greeks analytically.

    Returns
    -------
    dict with keys: delta, gamma, vega, theta, rho
        vega  : per 1% vol move (divided by 100)
        theta : per calendar day (divided by 365), negative for long options
        rho   : per 1% rate move (divided by 100)

    Returns dict of np.nan on invalid inputs — always safe to unpack.
    """
    nan_result = dict(delta=np.nan, gamma=np.nan,
                      vega=np.nan,  theta=np.nan, rho=np.nan)

    if T <= 0 or sigma <= 0 or spot <= 0 or K <= 0:
        return nan_result

    sqrt_T = np.sqrt(T)
    d1     = (np.log(spot / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2     = d1 - sigma * sqrt_T
    nd1    = norm.pdf(d1)
    disc   = np.exp(-r * T)

    # Delta
    delta = float(norm.cdf(d1)) if option_type == "call" else float(norm.cdf(d1) - 1.0)

    # Gamma (same for calls and puts)
    gamma = float(nd1 / (spot * sigma * sqrt_T))

    # Vega per 1% vol move
    vega = float(spot * sqrt_T * nd1 / 100.0)

    # Theta per calendar day
    if option_type == "call":
        theta_annual = (-(spot * nd1 * sigma) / (2.0 * sqrt_T)
                        - r * K * disc * norm.cdf(d2))
    else:
        theta_annual = (-(spot * nd1 * sigma) / (2.0 * sqrt_T)
                        + r * K * disc * norm.cdf(-d2))
    theta = float(theta_annual / 365.0)

    # Rho per 1% rate move
    if option_type == "call":
        rho = float(K * T * disc * norm.cdf(d2)  / 100.0)
    else:
        rho = float(-K * T * disc * norm.cdf(-d2) / 100.0)

    return dict(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


# ─────────────────────────────────────────────
# STANDALONE VEGA  (for surface notebook row-apply)
# ─────────────────────────────────────────────

def compute_vega(spot, K, T, r, sigma):
    """
    Black-Scholes Vega per unit vol move (not per 1%).

    Used as: dollar_mispricing = compute_vega(...) * iv_residual

    Returns 0.0 on invalid inputs.
    """
    if T <= 0 or sigma <= 0 or spot <= 0 or K <= 0:
        return 0.0
    sqrt_T = np.sqrt(T)
    d1     = (np.log(spot / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    return float(spot * sqrt_T * norm.pdf(d1))


# ─────────────────────────────────────────────
# VOL SURFACE FITTING & QUERY
# ─────────────────────────────────────────────

def fit_vol_surface(log_moneyness, T_col, iv_col,
                    n_mono=60, n_t=40,
                    mono_q=(0.05, 0.95)):
    """
    Fit a smooth 2D implied volatility surface from scattered data.

    Uses scipy.interpolate.griddata with cubic method, falling back
    to nearest-neighbour at boundaries where cubic cannot extrapolate.

    Parameters
    ----------
    log_moneyness : array-like  — ln(K/S) per contract
    T_col         : array-like  — time to expiry in years per contract
    iv_col        : array-like  — implied vol per contract
    n_mono        : int         — grid points along moneyness axis
    n_t           : int         — grid points along T axis
    mono_q        : tuple       — quantile bounds for moneyness grid
                                  avoids extrapolating into sparse tails

    Returns
    -------
    dict with keys:
        surface_iv  : 2D numpy array (n_t x n_mono) of interpolated IV
        mono_grid   : 1D array of log-moneyness grid points
        t_grid      : 1D array of T grid points
        points      : Nx2 array of input (log_moneyness, T) coordinates
        values      : N array of input IV values

    Raises
    ------
    ValueError if zero valid data points remain after NaN filtering.
    This usually means iv_calc was not computed across all expiries —
    see notebook 02 fix to compute IV for all expiries, not just demo_expiry.
    """
    lm = np.asarray(log_moneyness, dtype=float)
    t  = np.asarray(T_col,         dtype=float)
    iv = np.asarray(iv_col,        dtype=float)

    # Remove rows where any value is NaN or infinite
    mask = np.isfinite(lm) & np.isfinite(t) & np.isfinite(iv)
    lm, t, iv = lm[mask], t[mask], iv[mask]

    # Guard — surface cannot be fitted on empty data
    if len(lm) == 0:
        raise ValueError(
            "fit_vol_surface received 0 valid data points after NaN filtering.\n"
            "Most likely cause: iv_calc is NaN for all contracts in surf_calls.\n"
            "Fix: update notebook 02 to compute IV across ALL expiries, not just\n"
            "the single demo_expiry. Then re-run: bash run_all.sh --surface-only"
        )

    if len(lm) < 10:
        raise ValueError(
            f"fit_vol_surface received only {len(lm)} valid data points — too few "
            "to fit a reliable surface. Need at least 10. Check liquidity filters "
            "in notebook 05 Step 2 — the spread_ratio threshold may be too tight."
        )

    mono_grid   = np.linspace(np.quantile(lm, mono_q[0]),
                              np.quantile(lm, mono_q[1]), n_mono)
    t_grid      = np.linspace(t.min(), t.max(), n_t)
    MONO, T_MAT = np.meshgrid(mono_grid, t_grid)

    points = np.column_stack([lm, t])
    values = iv

    # Check T range — cubic Delaunay triangulation fails when all points
    # share the same T value (only one expiry in the data).
    t_range = t.max() - t.min()
    if t_range < 1e-6:
        raise ValueError(
            f"fit_vol_surface: all {len(t)} data points have the same T value "
            f"({t.min():.6f} years). Cannot fit a 2D surface from 1D data.\n"
            "This means only one expiry survived the liquidity filters in notebook 05.\n"
            "Fix: check notebook 05 Step 2 — the min-maturity filter (T >= 3/365) "
            "may be excluding all but the nearest expiry. Try relaxing it to T >= 1/365, "
            "or verify that opts_with_bs.parquet contains IV across multiple expiries "
            "by running: pd.read_parquet('data/opts_with_bs.parquet')['expiry'].value_counts()"
        )

    # Cubic interpolation — smooth surface through data points
    # Falls back to linear if Qhull triangulation fails (e.g. near-degenerate data)
    try:
        surf_cubic = griddata(points, values, (MONO, T_MAT), method="cubic")
    except Exception:
        # Qhull can fail on near-degenerate configurations — fall back to linear
        surf_cubic = griddata(points, values, (MONO, T_MAT), method="linear")

    # Nearest-neighbour fallback for any remaining NaN (boundary regions)
    surf_nn    = griddata(points, values, (MONO, T_MAT), method="nearest")
    surface_iv = np.where(np.isnan(surf_cubic), surf_nn, surf_cubic)

    return dict(
        surface_iv=surface_iv,
        mono_grid=mono_grid,
        t_grid=t_grid,
        points=points,
        values=values,
    )


def query_surface_iv(surface_result, log_moneyness_query, T_query):
    """
    Query a fitted vol surface at arbitrary (log_moneyness, T) coordinates.

    Parameters
    ----------
    surface_result      : dict returned by fit_vol_surface()
    log_moneyness_query : array-like of ln(K/S) values to query
    T_query             : array-like of T values to query

    Returns
    -------
    numpy array of interpolated IV values, same length as query inputs.
    Uses cubic interpolation with nearest-neighbour fallback for boundary points.
    """
    points = surface_result["points"]
    values = surface_result["values"]

    lm_q = np.asarray(log_moneyness_query, dtype=float)
    t_q  = np.asarray(T_query,             dtype=float)

    query_points = np.column_stack([lm_q, t_q])

    iv_cubic = griddata(points, values, query_points, method="cubic")
    iv_nn    = griddata(points, values, query_points, method="nearest")
    return np.where(np.isnan(iv_cubic), iv_nn, iv_cubic)
