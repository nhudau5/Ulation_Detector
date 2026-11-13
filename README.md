# Streamlit: Customer Feedback Sentiment

## Struktur
```
app.py
requirements.txt
.streamlit/config.toml
models/
  customer_feedback_sentiment_model.joblib   <- dari notebook
  metadata.json                              <- dari notebook (opsional)
```

## Jalankan lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
1. Push folder ini ke repo GitHub (termasuk `models/`).
2. Di share.streamlit.io: New app -> pilih repo, branch, main file `app.py`.
3. Pin `scikit-learn` di `requirements.txt` ke versi training sebelum deploy.

Catatan: file `.joblib` > 100 MB tidak bisa di-push biasa ke GitHub (pakai Git LFS).
