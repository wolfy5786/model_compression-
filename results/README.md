# Results

Baseline (FP32) outputs live here:

| File | Purpose |
|------|---------|
| `baseline_results.csv` | Flat metrics table for reuse |
| `baseline_results.json` | Full metrics + split metadata |
| `baseline_summary.md` | Human-readable summary |
| `baseline_results_*.json` | Timestamped snapshots |

Regenerate:

```powershell
.\.venv\Scripts\python.exe benchmarks\run_baselines.py
.\.venv\Scripts\python.exe benchmarks\run_baselines.py --models temponet --resume --temponet-epochs 15
```
