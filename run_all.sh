#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# run_all.sh — Execute all notebooks in dependency order
#
# Dependency graph:
#   01 → 02 → 03 → 04         (original pricing pipeline)
#   01 → 02 → 05 → 07         (vol surface + Greeks pipeline)
#
# Usage:
#   bash run_all.sh                  # run everything
#   bash run_all.sh --skip-mc        # skip notebook 04 (slow MC)
#   bash run_all.sh --surface-only   # run 01, 02, 05, 07 only
#   bash run_all.sh --greeks-only    # run 01, 02, 05, 07 only (alias)
# ─────────────────────────────────────────────────────────────────────

set -e  # stop immediately if any notebook fails

SKIP_MC=false
SURFACE_ONLY=false

for arg in "$@"; do
    case $arg in
        --skip-mc)                   SKIP_MC=true       ;;
        --surface-only|--greeks-only) SURFACE_ONLY=true ;;
    esac
done

TIMEOUT_STD=300
TIMEOUT_MC=600
TIMEOUT_SURF=400
TIMEOUT_GREEKS=400

run_nb() {
    local nb=$1
    local timeout=$2
    echo ""
    echo "──────────────────────────────────────"
    echo "Running $nb ..."
    echo "──────────────────────────────────────"
    jupyter nbconvert \
        --to notebook \
        --execute \
        --inplace \
        --ExecutePreprocessor.timeout=$timeout \
        "$nb"
    echo "✓  $nb complete"
}

# ── Phase 1: Data extraction (always runs) ────────────────────────────
run_nb 01_data_extraction.ipynb     $TIMEOUT_STD

# ── Phase 2: Black-Scholes & IV across all expiries (always runs) ─────
run_nb 02_black_scholes_iv.ipynb    $TIMEOUT_STD

if [ "$SURFACE_ONLY" = false ]; then
    # ── Phase 3: Binomial CRR (European + American) ───────────────────
    run_nb 03_binomial_crr.ipynb    $TIMEOUT_STD

    # ── Phase 4: Monte Carlo (optional skip) ──────────────────────────
    if [ "$SKIP_MC" = false ]; then
        run_nb 04_monte_carlo.ipynb $TIMEOUT_MC
    else
        echo ""
        echo "Skipping 04_monte_carlo.ipynb (--skip-mc flag set)"
    fi
fi

# ── Phase 5: Vol Surface, Arbitrage Checks & Mispricing ───────────────
run_nb 05_vol_surface.ipynb         $TIMEOUT_SURF

# ── Phase 6: Greeks (Delta, Gamma, Vega, Theta, Rho) ─────────────────
# Depends on 05 output: data/opts_with_surface.parquet
run_nb 07_greeks.ipynb              $TIMEOUT_GREEKS

echo ""
echo "══════════════════════════════════════"
echo "All notebooks completed successfully."
echo ""
echo "Parquet outputs:"
echo "  data/opts_clean.parquet"
echo "  data/opts_with_bs.parquet"
if [ "$SURFACE_ONLY" = false ]; then
echo "  data/opts_with_binomial.parquet"
if [ "$SKIP_MC" = false ]; then
echo "  data/opts_final.parquet"
fi
fi
echo "  data/opts_with_surface.parquet"
echo "  data/opts_with_greeks.parquet"
echo ""
echo "Chart outputs (plots/):"
echo "  spy_iv_smile_*.html"
echo "  binomial_eur_error_plot.html"
echo "  mc_convergence.html"
echo "  vol_surface_3d.html"
echo "  breeden_litzenberger_density.html"
echo "  iv_residual_heatmap.html"
echo "  dollar_mispricing_scatter.html"
echo "  trade_pnl_distribution.html"
echo "  pnl_attribution.html"
echo "  greeks_vs_strike.html"
echo "  delta_surface.html"
echo "  gamma_surface.html"
echo "  vega_surface.html"
echo "  theta_surface.html"
echo "  gamma_theta_tradeoff.html"
echo "  greeks_sensitivity.html"
echo "══════════════════════════════════════"
