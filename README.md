# Double/Debiased ML (DoubleML) Engine

Estimates heterogeneous treatment effects (CATE) of a macro shock (e.g., VIX spike) on ETF returns using Double/Debiased ML (DML) from `econml`. Controls for high‑dimensional confounders via cross‑fitting. Higher positive CATE indicates ETF that benefits most from the shock → overweight signal.

- **Treatment:** VIX change > 0.5σ above rolling mean
- **Features:** macro variables + market return
- **Estimator:** LinearDML (econml) with random forest for nuisance models
- **Windows:** 63, 252, 504, 1008, 2016 days (best per ETF)
- **Output:** top 3 ETFs per universe by CATE

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
