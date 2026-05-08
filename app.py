import re
import ast
import pickle
import joblib
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import streamlit as st

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Hashtag Recommender", layout="wide")

# =========================================================
# CONFIG
# =========================================================
BASE_DIR = Path(".")
DATA_PATH = BASE_DIR / "instagram_dataset_tfidf_ready.csv"
MODEL_DIR = BASE_DIR / "saved_models"

TOP_N_HASHTAGS = 100
MAX_FEATURES_TFIDF = 2000
MIN_DF = 10
MAX_DF = 0.85
SVM_C = 0.05
SEED = 42

RECENT_DAYS = 30
OLDER_DAYS = 90
TREND_MIN_FREQ = 3

# Strictly matching the updated model parameters
TREND_W_BASE = 0.70
TREND_W_COS = 0.20
TREND_W_TREND = 0.10
GENERIC_PENALTY_VAL = 0.00  

GENERIC_TAGS = {
    "#ad", "#love", "#instagood", "#photooftheday", "#photography",
    "#happy", "#happiness", "#life", "#beautiful", "#art", "#support"
}

# =========================================================
# HELPERS
# =========================================================
def safe_parse(x):
    try:
        return ast.literal_eval(x) if isinstance(x, str) else []
    except Exception:
        return []

def minmax_normalize_dict(d):
    if len(d) == 0:
        return {}
    values = np.array(list(d.values()), dtype=float)
    min_v = values.min()
    max_v = values.max()
    if max_v - min_v == 0:
        return {k: 0.0 for k in d}
    return {k: (v - min_v) / (max_v - min_v) for k, v in d.items()}

def minmax_col(series):
    series = series.astype(float)
    mn, mx = series.min(), series.max()
    if mx - mn == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mn) / (mx - mn)

def get_model_text(bundle, caption: str, category: str):
    category = str(category).strip().lower()
    caption = str(caption).strip().lower()
    return f"{category} {caption}" if bundle.get("use_category", False) else caption

# Make trend explanations punchy
def generate_trend_explanation(row):
    vel = row["velocity"]
    eng_growth = row["engagement_growth"]
    recent_ct = row["recent_count"]

    if vel >= 1.5 and eng_growth >= 1.2:
        return "🔥 Viral Spike"
    elif vel >= 1.5:
        return "📈 Gaining Momentum"
    elif eng_growth >= 1.5:
        return "💬 High Engagement"
    elif recent_ct >= 10: 
        return "⭐ Established Staple"
    else:
        return "" 

# =========================================================
# OPTIONAL MODEL LOADER: SBERT + LR
# =========================================================
@st.cache_resource
def load_sbert_lr_bundle(export_dir=MODEL_DIR):
    export_dir = Path(export_dir)
    clf_path = export_dir / "sbert_lr_clf.joblib"
    meta_path = export_dir / "sbert_lr_meta.pkl"

    if not clf_path.exists() or not meta_path.exists():
        return None

    clf = joblib.load(clf_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    embedder = SentenceTransformer(meta["model_name"])

    return {
        "name": "sbert_lr",
        "display_name": "SBERT + LR",
        "model": clf,
        "vectorizer": embedder,
        "mlb": meta["mlb"],
        "use_category": meta.get("use_category", True),
        "model_type": meta["model_type"],
    }

# =========================================================
# DATA LOADING
# =========================================================
@st.cache_data
def load_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    df["hashtags_list"] = df["hashtags_list"].apply(safe_parse)
    df["clean_caption"] = df["clean_caption"].fillna("")
    df["category"] = df["category"].fillna("unknown").astype(str).str.title().str.strip() 

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        df["timestamp"] = pd.NaT

    if "likes" not in df.columns: df["likes"] = 0
    if "comment_count" not in df.columns: df["comment_count"] = 0
    if "followers" not in df.columns: df["followers"] = 0

    return df

# =========================================================
# EXPERIMENT PREP
# =========================================================
def prepare_experiment_data(df: pd.DataFrame, top_n_hashtags=TOP_N_HASHTAGS):
    all_tags = [tag for tags in df["hashtags_list"] for tag in tags]
    top_tags = set([tag for tag, _ in Counter(all_tags).most_common(top_n_hashtags)])

    work_df = df.copy()
    work_df["hashtags_list"] = work_df["hashtags_list"].apply(lambda tags: [t for t in tags if t in top_tags])
    work_df = work_df[work_df["hashtags_list"].map(len) > 0].reset_index(drop=True)
    
    work_df["model_text"] = work_df["category"].str.lower() + " " + work_df["clean_caption"]

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(work_df["hashtags_list"])
    categories = work_df["category"].values

    from sklearn.model_selection import train_test_split
    train_df, temp_df, y_train, y_temp = train_test_split(
        work_df, y, test_size=0.2, random_state=SEED, stratify=categories
    )
    temp_categories = temp_df["category"].values
    val_df, test_df, y_val, y_test = train_test_split(
        temp_df, y_temp, test_size=0.5, random_state=SEED, stratify=temp_categories
    )

    return {
        "train_df": train_df.reset_index(drop=True),
        "y_train": y_train,
        "mlb": mlb,
        "use_category": True,
    }

# =========================================================
# MODEL TRAINING: SVM
# =========================================================
@st.cache_resource
def run_caption_svm(_train_text, _y_train, _mlb):
    tfidf = TfidfVectorizer(
        max_features=MAX_FEATURES_TFIDF,
        ngram_range=(1, 1),
        sublinear_tf=True,
        min_df=MIN_DF,
        max_df=MAX_DF,
    )
    X_train = tfidf.fit_transform(_train_text)
    base = LinearSVC(C=SVM_C, dual=False, max_iter=3000, random_state=SEED)
    model = OneVsRestClassifier(base)
    model.fit(X_train, _y_train)

    return {
        "name": "caption_svm",
        "display_name": "SVM",
        "model": model,
        "vectorizer": tfidf,
        "mlb": _mlb,
        "use_category": True,
        "model_type": "sklearn",
    }

# =========================================================
# TREND SCORING & CATEGORY TRACKING
# =========================================================
def prepare_trend_base_df(train_df):
    df = train_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    df["engagement_rate"] = (df["likes"].fillna(0) + df["comment_count"].fillna(0)) / (df["followers"].fillna(0) + 1)
    return df

def build_trend_score_table(train_df, recent_days=RECENT_DAYS, older_days=OLDER_DAYS, min_freq=TREND_MIN_FREQ):
    df = prepare_trend_base_df(train_df)

    if len(df) == 0:
        return pd.DataFrame(), {}

    max_time = df["timestamp"].max()
    recent_start = max_time - pd.Timedelta(days=recent_days)
    older_start = max_time - pd.Timedelta(days=older_days)

    recent_df = df[df["timestamp"] >= recent_start].copy()
    older_df = df[(df["timestamp"] >= older_start) & (df["timestamp"] < recent_start)].copy()

    recent_counts = Counter()
    older_counts = Counter()
    recent_eng_sum = defaultdict(float)
    older_eng_sum = defaultdict(float)
    recent_recency_sum = defaultdict(float)
    
    tag_categories = defaultdict(Counter)

    for _, row in recent_df.iterrows():
        tags = row["hashtags_list"]
        eng = float(row["engagement_rate"])
        age_days = max((max_time - row["timestamp"]).days, 0)
        recency_weight = 1 / (1 + age_days)
        cat = str(row["category"])

        for tag in tags:
            recent_counts[tag] += 1
            recent_eng_sum[tag] += eng
            recent_recency_sum[tag] += recency_weight
            tag_categories[tag][cat] += 1

    for _, row in older_df.iterrows():
        tags = row["hashtags_list"]
        eng = float(row["engagement_rate"])
        cat = str(row["category"])

        for tag in tags:
            older_counts[tag] += 1
            older_eng_sum[tag] += eng
            tag_categories[tag][cat] += 1

    all_tags = set(recent_counts.keys()) | set(older_counts.keys())
    rows = []

    for tag in all_tags:
        r_count = recent_counts.get(tag, 0)
        o_count = older_counts.get(tag, 0)
        total_freq = r_count + o_count
        if total_freq < min_freq:
            continue

        r_avg_eng = recent_eng_sum[tag] / r_count if r_count > 0 else 0.0
        o_avg_eng = older_eng_sum[tag] / o_count if o_count > 0 else 0.0

        velocity = (r_count + 1) / (o_count + 1)
        engagement_growth = (r_avg_eng + 1) / (o_avg_eng + 1)
        recency_score = recent_recency_sum.get(tag, 0.0)
        
        primary_domain = tag_categories[tag].most_common(1)[0][0] if tag_categories[tag] else "General"

        rows.append({
            "tag": tag,
            "category": primary_domain,
            "recent_count": r_count,
            "older_count": o_count,
            "velocity": velocity,
            "engagement_growth": engagement_growth,
            "recency_score": recency_score,
        })

    trend_df = pd.DataFrame(rows)

    if len(trend_df) == 0:
        return pd.DataFrame(), {}

    trend_df["velocity_norm"] = minmax_col(trend_df["velocity"])
    trend_df["engagement_growth_norm"] = minmax_col(trend_df["engagement_growth"])
    trend_df["recency_norm"] = minmax_col(trend_df["recency_score"])

    trend_df["trend_score"] = (
        0.45 * trend_df["velocity_norm"] +
        0.35 * trend_df["engagement_growth_norm"] +
        0.20 * trend_df["recency_norm"]
    )
    
    trend_df["trend_reason"] = trend_df.apply(generate_trend_explanation, axis=1)
    trend_df = trend_df.sort_values("trend_score", ascending=False).reset_index(drop=True)
    trend_score_dict = dict(zip(trend_df["tag"], trend_df["trend_score"]))
    
    return trend_df, trend_score_dict

# =========================================================
# CORE RERANKING LOGIC
# =========================================================
def get_bundle_scores(bundle, caption: str, category: str):
    model_text = get_model_text(bundle, caption, category)

    if bundle["model_type"] == "sklearn":
        X = bundle["vectorizer"].transform([model_text])
        scores = bundle["model"].decision_function(X)[0]
    elif bundle["model_type"] in ["embedding_clf", "embedding_clf_proba"]:
        X = bundle["vectorizer"].encode([model_text], convert_to_numpy=True, normalize_embeddings=True)
        scores = bundle["model"].predict_proba(X)[0] if bundle["model_type"] == "embedding_clf_proba" else bundle["model"].decision_function(X)[0]
    else:
        raise ValueError(f"Unsupported model_type: {bundle['model_type']}")

    return scores

def compute_semantic_cos_score(bundle, caption: str, tag: str):
    if not caption.strip():
        return 0.0
    
    if bundle["model_type"] in ["embedding_clf", "embedding_clf_proba"]:
        cap_emb = bundle["vectorizer"].encode([caption])
        tag_emb = bundle["vectorizer"].encode([tag])
        return float(cosine_similarity(cap_emb, tag_emb)[0][0])
    else:
        cap_vec = bundle["vectorizer"].transform([caption])
        tag_vec = bundle["vectorizer"].transform([tag])
        return float(cosine_similarity(cap_vec, tag_vec)[0][0])

def raw_predict(bundle, caption: str, category: str, candidate_pool=50):
    scores = get_bundle_scores(bundle, caption, category)
    cand_idx = np.argsort(scores)[::-1][:candidate_pool]
    cand_tags = [bundle["mlb"].classes_[i] for i in cand_idx]
    cand_scores = {tag: float(scores[i]) for tag, i in zip(cand_tags, cand_idx)}
    return cand_scores

def strict_rerank_pipeline(bundle, trend_score_dict, caption: str, category: str, candidate_pool=50, top_k=5):
    cand_scores = raw_predict(bundle, caption, category, candidate_pool=candidate_pool)
    cand_norm = minmax_normalize_dict(cand_scores)

    rows = []
    for tag in cand_scores.keys():
        base_score = cand_norm.get(tag, 0.0)
        trend_score = float(trend_score_dict.get(tag, 0.0))
        cos_score = compute_semantic_cos_score(bundle, caption, tag)
        penalty = GENERIC_PENALTY_VAL if tag in GENERIC_TAGS else 0.0

        final_score = (
            (TREND_W_BASE * base_score) +
            (TREND_W_TREND * trend_score) +
            (TREND_W_COS * cos_score) -
            penalty
        )

        rows.append({
            "tag": tag,
            "final_score": float(final_score)
        })

    rows = sorted(rows, key=lambda x: x["final_score"], reverse=True)
    return [r["tag"] for r in rows[:top_k]]

# =========================================================
# APP BUILDER & RUNNER
# =========================================================
@st.cache_resource
def build_app_objects(csv_path: str):
    df = load_dataset(csv_path)
    exp = prepare_experiment_data(df, top_n_hashtags=TOP_N_HASHTAGS)

    svm_bundle = run_caption_svm(
        _train_text=exp["train_df"]["model_text"],
        _y_train=exp["y_train"],
        _mlb=exp["mlb"]
    )

    trend_df, trend_score_dict = build_trend_score_table(exp["train_df"])

    try:
        sbert_lr_bundle = load_sbert_lr_bundle()
    except Exception:
        sbert_lr_bundle = None

    models = {"SVM (Base)": svm_bundle}
    if sbert_lr_bundle is not None:
        models["SBERT + LR"] = sbert_lr_bundle

    return {
        "exp": exp,
        "models": models,
        "trend_df": trend_df,
        "trend_score_dict": trend_score_dict,
    }

def run_model_family(system, family_name, caption, category, top_k=5, candidate_pool=50):
    trend_score_dict = system["trend_score_dict"]

    if family_name == "SVM (Base)":
        bundle = system["models"]["SVM (Base)"]
    elif family_name == "SBERT + LR":
        bundle = system["models"]["SBERT + LR"]
    else:
        raise ValueError(f"Unknown family: {family_name}")

    final_tags = strict_rerank_pipeline(bundle, trend_score_dict, caption, category, candidate_pool, top_k)
    return final_tags

# =========================================================
# UI
# =========================================================
st.title("Hashtag Recommender & Trends")

system = build_app_objects(str(DATA_PATH))
categories = sorted(system["exp"]["train_df"]["category"].dropna().unique().tolist())

family_options = ["SVM (Base)"]
if "SBERT + LR" in system["models"]:
    family_options.append("SBERT + LR")

# --- SECTION 1: HASHTAG GENERATOR ---
st.subheader("✍️ Draft Your Post")

col_cat, col_fam = st.columns([1, 1])
with col_cat:
    default_idx = categories.index("Fitness") if "Fitness" in categories else 0
    category = st.selectbox("Select Post Category", categories, index=default_idx)
with col_fam:
    selected_family = st.selectbox("Select Intelligence Model", family_options, index=0)

caption = st.text_area("Post Caption", value="Having a great morning workout at the gym, feeling strong today!", height=100)

with st.expander("⚙️ Advanced Settings"):
    col1, col2 = st.columns([1, 1])
    with col1:
        top_k = st.slider("Number of Hashtags to Output", min_value=3, max_value=15, value=5)
    with col2:
        candidate_pool = st.slider("Internal Search Space", min_value=10, max_value=100, value=50, step=5)

run_btn = st.button("Generate Recommendations", type="primary", use_container_width=True)

if run_btn:
    if not caption.strip():
        st.warning("Please enter a caption.")
    else:
        with st.spinner("Analyzing semantics and current trends..."):
            final_tags = run_model_family(system, selected_family, caption, category, top_k, candidate_pool)

        st.success("Recommendations Generated!")
        
        clean_tags = [f"#{str(t).replace('#', '')}" for t in final_tags]
        st.code(" ".join(clean_tags), language="markdown")

st.divider()

# --- SECTION 2: GLOBAL TRENDS ---
st.subheader("🔥 Platform Trends Dashboard")
st.caption(f"Filter and discover which topics gained traction over the last {RECENT_DAYS} days.")

trend_df = system["trend_df"]

if len(trend_df) > 0:
    # --- Category Filter ---
    all_trend_cats = ["All Categories"] + sorted(trend_df["category"].dropna().unique().tolist())
    selected_trend_cat = st.selectbox("Filter leaderboard by category:", all_trend_cats)
    
    if selected_trend_cat != "All Categories":
        display_df = trend_df[trend_df["category"] == selected_trend_cat].copy()
    else:
        display_df = trend_df.copy()

    st.markdown(f"##### 🏆 Top 3 in {selected_trend_cat}")
    top_3 = display_df.head(3)
    cols = st.columns(3)
    
    for i, (_, row) in enumerate(top_3.iterrows()):
        with cols[i]:
            tag_name = f"#{str(row['tag']).replace('#', '')}"
            status_label = row['trend_reason'] if row['trend_reason'] else f"📌 {row['category']}"
            
            st.metric(
                label=status_label, 
                value=tag_name, 
                delta=f"{int(row['recent_count'])} uses (Last {RECENT_DAYS} Days)", 
                delta_color="normal"
            )

    st.markdown("##### 📊 Full Analytical Leaderboard")
    
    # 1. Translate raw math into user-friendly metrics
    clean_trend_df = pd.DataFrame()
    clean_trend_df["Hashtag"] = display_df["tag"].apply(lambda x: f"#{str(x).replace('#', '')}")
    clean_trend_df["Primary Category"] = display_df["category"]
    
    # Convert multipliers (e.g., 1.04) into percentage change (e.g., +4%)
    clean_trend_df["Velocity"] = (display_df["velocity"] - 1.0) * 100
    clean_trend_df["Engagement"] = (display_df["engagement_growth"] - 1.0) * 100
    
    # Convert raw 0.0-1.0 score into a 0-100 "Index"
    clean_trend_df["Trend Index"] = (display_df["trend_score"] * 100).astype(int)
    
    clean_trend_df["Volume"] = display_df["recent_count"].astype(int)
    max_volume = int(clean_trend_df["Volume"].max()) if not clean_trend_df.empty else 10

    # 2. Use Streamlit column configs to format the UI beautifully
    st.dataframe(
        clean_trend_df.head(50),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Hashtag": st.column_config.TextColumn("Trending Hashtag", width="medium"),
            "Primary Category": st.column_config.TextColumn("Category"),
            "Velocity": st.column_config.NumberColumn(
                "Velocity", 
                help="Percentage change in usage speed compared to the previous period.",
                format="%+d%%"  # Automatically formats as +5% or -20%
            ),
            "Engagement": st.column_config.NumberColumn(
                "Eng. Growth", 
                help="Percentage change in audience engagement (likes/comments).",
                format="%+d%%"
            ),
            "Trend Index": st.column_config.ProgressColumn(
                "Trend Index", 
                help="Overall algorithmic trend strength (0-100 scale).",
                format="%d",
                min_value=0,
                max_value=100
            ),
            "Volume": st.column_config.ProgressColumn(
                f"Volume (Last {RECENT_DAYS} Days)",
                help=f"Total raw uses in the last {RECENT_DAYS} days.",
                format="%d uses",
                min_value=0,
                max_value=max_volume,
            ),
        }
    )
else:
    st.info("No trend data available. Make sure your dataset includes timestamps and engagement metrics.")
