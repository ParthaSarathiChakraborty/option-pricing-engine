#!/bin/bash
set -e  # stops the script immediately if any notebook fails

echo "Running 01 — Data Extraction..."
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 01_data_extraction.ipynb

echo "Running 02 — Black-Scholes & IV..."
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 02_black_scholes_iv.ipynb

echo "Running 03 — Binomial CRR..."
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 03_binomial_crr.ipynb

echo "Running 04 — Monte Carlo..."
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 04_monte_carlo.ipynb

echo "All notebooks completed successfully."
