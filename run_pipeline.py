#!/usr/bin/env python
"""
run_pipeline.py
===============
Business Insights & Sales Forecasting Tool
Master pipeline runner for Deliverable 2

Steps:
  1. Clean raw data          (scripts/data_cleaning.py — must have been run)
  2. Build churn features    (src/features/churn_features.py)
  3. Train churn model       (src/training/train_churn.py)
  4. Evaluate churn model    (src/evaluation/evaluate_churn.py)

Usage:
  python run_pipeline.py [--skip-cleaning]
"""

import argparse
import subprocess
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_step(label: str, module: str):
    print(f"\n{'='*60}")
    print(f"  STEP: {label}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, "-m", module],
        cwd=BASE_DIR,
    )
    if result.returncode != 0:
        print(f"\n❌ Step failed: {label}")
        sys.exit(1)
    print(f"\n✅ {label} — done")


def main():
    parser = argparse.ArgumentParser(description="BISFT Deliverable 2 Pipeline")
    parser.add_argument(
        "--skip-cleaning", action="store_true",
        help="Skip data cleaning step (if already run)"
    )
    args = parser.parse_args()

    if not args.skip_cleaning:
        run_step("Data Cleaning",       "scripts.data_cleaning")

    run_step("Feature Engineering",     "src.features.churn_features")
    run_step("Model Training & HP Search", "src.training.train_churn")
    run_step("Model Evaluation",        "src.evaluation.evaluate_churn")

    print("\n" + "="*60)
    print("  🎉 Full pipeline complete!")
    print("  → Metrics:  reports/tables/final_metrics.csv")
    print("  → Figures:  reports/figures/")
    print("  → Models:   models/churn/")
    print("  → Eval:     evaluation/churn/")
    print("="*60)


if __name__ == "__main__":
    main()
