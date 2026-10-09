# Hybrid Attack Detection and Automated Response (Simulation)

Simulates window-level appliance telemetry (benign, benign-admin, recon, exploitation,
persistence, credential/session abuse, and a novel zero-day-like variant), then compares
four detectors: rules, Isolation Forest, Random Forest, and a hybrid risk score. It also
simulates automated vs. manual containment time.

**GitHub:** https://github.com/GarvB09/cybersecurity_proj

## Files

| File | Purpose |
|---|---|
| `detect.py` | Source: data generation, models, metrics, figures |
| `detect.exe` | Windows executable built from `detect.py` (PyInstaller, no Python needed) |
| `requirements.txt` | Python dependencies |
| `results.json`, `fig_*.png` | Sample output from a run |

## How to run

### Option A: executable (Windows)
1. Open a terminal in the project folder.
2. Run `detect.exe` (takes about 1-5 minutes; the first start unpacks itself).
3. Results print to the console. `results.json` and `fig_roc.png`, `fig_f1.png`,
   `fig_features.png`, `fig_response.png` are written to the current folder.

### Option B: from source
```
python -m pip install -r requirements.txt
python detect.py
```
Requires Python 3.9+. The random seeds are fixed, so results are reproducible.

## Expected output (summary)
- Known attacks: Random Forest F1 about 0.905, hybrid F1 about 0.893 (rules about 0.41, Isolation Forest about 0.49).
- Novel variant: Random Forest F1 about 0.842, hybrid about 0.816.
- Containment: 22 min automated vs. 625 min manual.

## Rebuild the exe
```
python -m pip install pyinstaller
python -m PyInstaller --onefile --name detect detect.py
```
The exe appears in `dist/`.

