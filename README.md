
# Streamlit ML App — Iris Example

Complete ML pipeline from training to deployment (Iris classification).

## 1) Setup (Windows PowerShell)
```pwsh
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

**Ubuntu/macOS**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Train the model
```bash
python train.py
```
Artifacts produced: `model.pkl`, `metrics.json`, `model_compare.json`, `data/dataset.csv`, `data/test.csv`, `data/confusion_matrix.png`, `model_info.json`.

## 3) Run locally
```bash
streamlit run app.py
```

## 4) Deploy to Streamlit Cloud
- Push this project to a **public GitHub repo**.
- Go to https://share.streamlit.io, connect your GitHub, pick the repo and the `main` branch.
- App entry point: `app.py`.
- Make sure `requirements.txt` is present.

## 5) Adapting to another dataset
- Replace `train.py` to load your dataset (e.g., Titanic) and set `feature_names` + `target` consistently.
- Re-run training to regenerate `model.pkl` & metrics.
- Update text and widgets in `app.py` (Predict page) if your features are not purely numeric.

