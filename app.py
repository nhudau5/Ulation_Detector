"""Frontend Streamlit untuk klasifikasi sentimen customer feedback (Bahasa Indonesia).

Membutuhkan file hasil notebook:
    models/customer_feedback_sentiment_model.joblib
    models/metadata.json   (opsional, untuk tab "Tentang Model")
"""

import json
import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "customer_feedback_sentiment_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "metadata.json"

MAX_CHARS = 2000
MAX_BATCH_ROWS = 5000
# Heuristik (bukan hasil kalibrasi): skor SVM di dalam rentang ini dianggap dekat batas keputusan.
BORDERLINE_MARGIN = 0.3

# ---------------------------------------------------------------------------
# Preprocessing: HARUS identik dengan notebook, kalau tidak hasil prediksi bergeser.
# ---------------------------------------------------------------------------
NORMALIZATION_DICT = {
    "bgt": "banget", "yg": "yang", "yng": "yang", "dgn": "dengan", "dr": "dari",
    "tp": "tetapi", "tpi": "tetapi", "krn": "karena", "karna": "karena",
    "gk": "tidak", "ga": "tidak", "gak": "tidak", "ngga": "tidak", "nggak": "tidak",
    "tdk": "tidak", "tak": "tidak", "blm": "belum", "udh": "sudah", "sdh": "sudah",
    "dah": "sudah", "sm": "sama", "spt": "seperti", "aja": "saja", "kalo": "kalau",
    "kl": "kalau", "jg": "juga", "jd": "jadi", "bkn": "bukan", "krg": "kurang",
    "hrs": "harus",
}

STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "ini", "itu", "pada",
    "adalah", "sebagai", "karena", "atau", "juga", "sudah", "akan", "ada", "dalam",
    "saya", "aku", "kami", "kita", "mereka", "dia", "nya", "saja", "lebih",
    "sangat", "bisa", "dapat", "oleh", "jadi", "agar", "buat", "seperti", "sama",
    "kalau", "ketika", "saat",
} - {"tidak", "bukan", "belum", "kurang", "bagus", "buruk"}


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"#[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-zA-ZÀ-ÿ\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text):
    return " ".join(NORMALIZATION_DICT.get(t, t) for t in text.split())


def remove_stopwords(text):
    return " ".join(t for t in text.split() if t not in STOPWORDS)


def preprocess(text):
    return remove_stopwords(normalize_text(clean_text(text)))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model...")
def load_bundle():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return None


def score_texts(bundle, processed_texts):
    """Return (labels, scores). Skor = probabilitas kelas Positive jika tersedia,
    kalau tidak, decision_function (mis. Linear SVM)."""
    model, tfidf = bundle["model"], bundle["tfidf"]
    X = tfidf.transform(processed_texts)
    labels = model.predict(X)
    if hasattr(model, "predict_proba"):
        pos_idx = list(model.classes_).index("Positive")
        return labels, model.predict_proba(X)[:, pos_idx], "proba"
    if hasattr(model, "decision_function"):
        return labels, model.decision_function(X), "margin"
    return labels, [None] * len(labels), "none"


def predict_one(bundle, text):
    processed = preprocess(text)
    if not processed:
        return None
    labels, scores, kind = score_texts(bundle, [processed])
    return {"processed": processed, "label": labels[0], "score": scores[0], "kind": kind}


def predict_many(bundle, texts):
    processed = [preprocess(t) for t in texts]
    valid_idx = [i for i, p in enumerate(processed) if p]
    out = pd.DataFrame({"processed_text": processed, "prediction": None, "score": None})
    if valid_idx:
        labels, scores, _ = score_texts(bundle, [processed[i] for i in valid_idx])
        out.loc[valid_idx, "prediction"] = list(labels)
        out.loc[valid_idx, "score"] = [float(s) for s in scores]
    out["prediction"] = out["prediction"].fillna("Kosong setelah preprocessing")
    return out


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Analisis Sentimen Feedback", page_icon="💬", layout="centered")
st.title("💬 Analisis Sentimen Customer Feedback")
st.caption("Klasifikasi Positive / Negative untuk ulasan berbahasa Indonesia (TF-IDF + model ML).")

if not MODEL_PATH.exists():
    st.error(
        f"File model tidak ditemukan: `{MODEL_PATH.relative_to(BASE_DIR)}`.\n\n"
        "Jalankan notebook sampai sel *Save Model dan TF-IDF*, lalu salin folder "
        "`models/` ke samping `app.py`."
    )
    st.stop()

bundle = load_bundle()
EXAMPLES = {
    "— pilih contoh —": "",
    "Positif": "Barangnya bagus banget dan pengiriman cepat",
    "Negatif": "Barang rusak dan sangat mengecewakan",
    "Singkatan / slang": "gak sesuai deskripsi, kecewa bgt udh nunggu lama",
}

tab_single, tab_batch, tab_about = st.tabs(["Satu ulasan", "Batch CSV", "Tentang model"])

# --- Tab 1: satu ulasan -----------------------------------------------------
with tab_single:
    example = st.selectbox("Contoh cepat", list(EXAMPLES.keys()))
    text = st.text_area(
        "Tulis feedback pelanggan",
        value=EXAMPLES[example],
        max_chars=MAX_CHARS,
        height=140,
        placeholder="Contoh: Barangnya bagus banget dan pengiriman cepat",
    )

    if st.button("Analisis", type="primary", disabled=not text.strip()):
        result = predict_one(bundle, text)
        if result is None:
            st.warning(
                "Teks kosong setelah preprocessing (hanya angka, simbol, atau stopword). "
                "Coba tulis kalimat yang lebih lengkap."
            )
        else:
            label, score, kind = result["label"], result["score"], result["kind"]
            (st.success if label == "Positive" else st.error)(f"**{label}**")

            if kind == "proba":
                st.metric("Probabilitas Positive", f"{score:.1%}")
            elif kind == "margin":
                st.metric("Skor keputusan (Positive > 0)", f"{score:+.3f}")
                st.caption(
                    "Ini jarak ke batas keputusan SVM, **bukan probabilitas**. "
                    "Semakin jauh dari 0, semakin tegas prediksinya."
                )
                if abs(score) < BORDERLINE_MARGIN:
                    st.warning("Skor dekat batas keputusan, hasil ini kurang stabil. Cek manual.")

            with st.expander("Teks setelah preprocessing"):
                st.code(result["processed"])

# --- Tab 2: batch CSV ---------------------------------------------------------
with tab_batch:
    st.write(f"Unggah CSV berisi kolom teks ulasan (maks. {MAX_BATCH_ROWS:,} baris).")
    uploaded = st.file_uploader("File CSV", type="csv")

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as exc:  # noqa: BLE001
            st.error(f"CSV tidak bisa dibaca: {exc}")
            st.stop()

        if df.empty:
            st.warning("CSV kosong.")
            st.stop()

        default_idx = list(df.columns).index("reviewContent") if "reviewContent" in df.columns else 0
        col = st.selectbox("Kolom teks", df.columns, index=default_idx)

        if len(df) > MAX_BATCH_ROWS:
            st.info(f"Hanya {MAX_BATCH_ROWS:,} baris pertama yang diproses (dari {len(df):,}).")
            df = df.head(MAX_BATCH_ROWS)

        if st.button("Prediksi semua", type="primary"):
            with st.spinner("Memproses..."):
                preds = predict_many(bundle, df[col].fillna("").astype(str).tolist())
            result_df = pd.concat([df.reset_index(drop=True), preds], axis=1)

            counts = result_df["prediction"].value_counts()
            st.bar_chart(counts)
            st.dataframe(result_df, use_container_width=True, height=320)
            st.download_button(
                "Unduh hasil (CSV)",
                result_df.to_csv(index=False).encode("utf-8"),
                file_name="hasil_sentimen.csv",
                mime="text/csv",
            )

# --- Tab 3: tentang model -------------------------------------------------------
with tab_about:
    meta = load_metadata()
    st.write(f"**Model:** {bundle.get('best_model_name', type(bundle['model']).__name__)}")

    if meta:
        st.write(f"**Jumlah data latih+uji:** {meta.get('dataset_rows', '?'):,}")
        metrics = {k: v for k, v in meta.get("best_metrics", {}).items() if k != "Model"}
        if metrics:
            st.write("**Metrik pada test set** (kelas Positive sebagai kelas target):")
            cols = st.columns(len(metrics))
            for c, (k, v) in zip(cols, metrics.items()):
                c.metric(k, f"{float(v):.3f}")

    st.markdown(
        """
**Batasan yang perlu diketahui**

- Data sangat tidak seimbang (sekitar 90% Positive). Akurasi terlihat tinggi, tetapi
  *balanced accuracy* jauh lebih rendah. Model lebih sering salah pada ulasan **Negative**
  (diprediksi Positive) daripada sebaliknya.
- Label berasal dari rating (1–2 = Negative, 4–5 = Positive; rating 3 dibuang), bukan
  anotasi manusia atas teks. Ulasan berisi pertanyaan atau keluhan netral bisa salah label.
- Hanya Bahasa Indonesia; kamus normalisasi slang terbatas (~30 kata).
- Tidak ada kelas Netral.
        """
    )
