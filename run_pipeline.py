"""
run_pipeline.py
----------------
Convenience entry point that runs the full pipeline end to end:
  feature engineering -> train + tune 3 models -> evaluate the winner
  -> save all charts/metrics to reports/

Run:
    python run_pipeline.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(script: str):
    print(f"\n{'=' * 70}\nRunning {script}\n{'=' * 70}")
    result = subprocess.run([sys.executable, str(ROOT / "src" / script)], cwd=str(ROOT / "src"))
    if result.returncode != 0:
        sys.exit(result.returncode)


if __name__ == "__main__":
    run("train.py")
    run("evaluate.py")
    print("\nPipeline complete. Launch the app with: streamlit run app.py")
