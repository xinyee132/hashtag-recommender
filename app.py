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
# CONFIG
# =========================================================
BASE_DIR = Path(".")
DATA_PATH = BASE_DIR / "instagram_dataset_tfidf_ready.csv"
MODEL_DIR = BASE_DIR / "saved_models"

TOP_N_HASHTAGS = 400
MAX_FEATURES_TFIDF = 2000
MIN_DF = 10
MAX_DF = 0.85
SVM_C = 0.05
SEED = 42

RECENT_DAYS = 30
OLDER_DAYS = 90
TREND_MIN_FREQ = 3

W_BASE = 0.35
W_SEM = 0.25
W_TREND = 0.15
W_CAT = 0.15
W_LEX = 0.10
GENERIC_PENALTY = 0.12

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

def tokenize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()

def hashtag_token(tag):
    tag = str(tag).lower().replace("#", "")
    tag = re.sub(r"[_\-]", " ", tag)
    tag = re.sub(r"[^a-z0-9\s]", "", tag)
    tag = re.sub(r"\s+", " ", tag).strip()
    return tag

def lexical_similarity(caption, tag):
    words = set(tokenize(caption))
    token = hashtag_token(tag)

    if token in words:
        return 1.0

    for w in words:
        if token in w or w in token:
            return 0.6

    return 0.0

def minmax_normalize_dict(d):
    if not d:
        return {}
    values = np.array(list(d.values()), dtype=float)
    mn, mx = values.min(), values.max()
    if mx - mn == 0:
        return {k: 0.0 for k in d}
    return {k: (v - mn) / (mx - mn) for k, v in d.items()}

def minmax_col(series):
    series = series.astype(float)
    mn, mx = series.min(), series.max()
    if mx - mn == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mn) / (mx - mn)

# =========================================================
# DATA
# =========================================================
@st.cache_data
def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df["hashtags_list"] = df["hashtags_list"].apply(safe_parse)
    df["clean_caption"] = df["clean_caption"].fillna("")
    df["category"] = df["category"].fillna("unknown").astype(str).str.lower().str.strip()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        df["timestamp"] = pd.NaT

    if "likes" not in df.columns:
        df["likes"] = 0
    if "comment_count" not in df.columns:
        df["comment_count"] = 0

    return df

def prepare_experiment_data(df, use_category=True, top_n_hashtags=TOP_N_HASHTAGS):
    all_tags = [tag for tags in df["hashtags_list"] for tag in tags]
    top_tags = set([tag for tag, _ in Counter(all_tags).most_common(top_n_hashtags)])

    work_df = df.copy()
    work_df["hashtags_list"] = work_df["hashtags_list"].apply(lambda tags: [t for t in tags if t in top_tags])
    work_df = work_df[work_df["hashtags_list"].map(len) > 0].reset_index(drop=True)

    if use_category:
        work_df["model_text"] = work_df["category"] + " " + work_df["clean_caption"]
    else:
        work_df["model_text"] = work_df["clean_caption"]

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(work_df["hashtags_list"])

    from sklearn.model_selection import train_test_split

    train_df, temp_df, y_train, y_temp = train_test_split(
        work_df,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=work_df["category"].values
    )

    val_df, test_df, y_val, y_test = train_test_split(
        temp_df,
        y_temp,
        test_size=0.5,
        random_state=SEED,
        stratify=temp_df["category"].values
    )

    return {
        "train_df": train_df.reset_index(drop=True),
        "val_df": val_df.reset_index(drop=True),
        "test_df": test_df.reset_index(drop=True),
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "mlb": mlb,
        "use_category": use_category
    }

# =========================================================
# MODEL LOADERS
# =========================================================
@st.cache_resource
def load_svm_bundle(_exp):
    tfidf = TfidfVectorizer(
        max_features=MAX_FEATURES_TFIDF,
        ngram_range=(1, 1),
        sublinear_tf=True,
        min_df=MIN_DF,
        max_df=MAX_DF,
    )
    X_train = tfidf.fit_transform(_exp["train_df"]["model_text"])

    base = LinearSVC(C=SVM_C, dual=False, max_iter=3000, random_state=SEED)
    model = OneVsRestClassifier(base)
    model.fit(X_train, _exp["y_train"])

    return {
        "model": model,
        "vectorizer": tfidf,
        "mlb": _exp["mlb"],
        "use_category": True,
    }

@st.cache_resource
def load_sbert_bundle(export_dir=MODEL_DIR):
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
        "model": clf,
        "vectorizer": embedder,
        "mlb": meta["mlb"],
        "use_category": meta["use_category"],
        "model_type": meta["model_type"]
    }

# =========================================================
# SUPPORT TABLES
# =========================================================
def build_category_affinity(train_df):
    category_tag_counts = defaultdict(Counter)

    for _, row in train_df.iterrows():
        cat = str(row["category"]).lower().strip()
        for tag in row["hashtags_list"]:
            category_tag_counts[cat][tag] += 1

    affinity = defaultdict(dict)
    for cat, counter in category_tag_counts.items():
        total = sum(counter.values())
        for tag, cnt in counter.items():
            affinity[cat][tag] = cnt / total if total > 0 else 0.0
    return affinity

def prepare_trend_base_df(train_df):
    df = train_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    df["engagement"] = df["likes"].fillna(0) + df["comment_count"].fillna(0)
    return df

def build_trend_score_table(train_df, recent_days=RECENT_DAYS, older_days=OLDER_DAYS, min_freq=TREND_MIN_FREQ):
    df = prepare_trend_base_df(train_df)
    if len(df) == 0:
        return pd.DataFrame(columns=["tag", "trend_score"]), {}

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

    for _, row in recent_df.iterrows():
        eng = float(row["engagement"])
        age_days = max((max_time - row["timestamp"]).days, 0)
        recency_weight = 1 / (1 + age_days)

        for tag in row["hashtags_list"]:
            recent_counts[tag] += 1
            recent_eng_sum[tag] += eng
            recent_recency_sum[tag] += recency_weight

    for _, row in older_df.iterrows():
        eng = float(row["engagement"])
        for tag in row["hashtags_list"]:
            older_counts[tag] += 1
            older_eng_sum[tag] += eng

    rows = []
    all_tags = set(recent_counts.keys()) | set(older_counts.keys())
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

        rows.append({
            "tag": tag,
            "velocity": velocity,
            "engagement_growth": engagement_growth,
            "recency_score": recency_score
        })

    trend_df = pd.DataFrame(rows)
    if len(trend_df) == 0:
        return pd.DataFrame(columns=["tag", "trend_score"]), {}

    trend_df["velocity_norm"] = minmax_col(trend_df["velocity"])
    trend_df["engagement_growth_norm"] = minmax_col(trend_df["engagement_growth"])
    trend_df["recency_norm"] = minmax_col(trend_df["recency_score"])

    trend_df["trend_score"] = (
        0.45 * trend_df["velocity_norm"] +
        0.35 * trend_df["engagement_growth_norm"] +
        0.20 * trend_df["recency_norm"]
    )

    trend_score_dict = dict(zip(trend_df["tag"], trend_df["trend_score"]))
    return trend_df, trend_score_dict

def build_hashtag_prototypes(train_df):
    hashtag_caption_tokens = defaultdict(Counter)
    hashtag_category_counts = defaultdict(Counter)

    for _, row in train_df.iterrows():
        cat = row["category"]
        tokens = tokenize(row["clean_caption"])
        for tag in row["hashtags_list"]:
            hashtag_category_counts[tag][cat] += 1
            hashtag_caption_tokens[tag].update(tokens)

    all_tags = sorted(set(tag for tags in train_df["hashtags_list"] for tag in tags))
    prototypes = {}
    for tag in all_tags:
        token = hashtag_token(tag)
        top_words = [w for w, _ in hashtag_caption_tokens[tag].most_common(12)]
        top_cats = [c for c, _ in hashtag_category_counts[tag].most_common(2)]
        prototypes[tag] = " ".join([token] + top_cats + top_words)

    return all_tags, prototypes

@st.cache_resource
def build_sbert_semantic_index(train_df):
    all_tags, prototypes = build_hashtag_prototypes(train_df)
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    hashtag_embeddings = embedder.encode(
        [prototypes[t] for t in all_tags],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return embedder, all_tags, hashtag_embeddings

# =========================================================
# SCORING
# =========================================================
def get_model_text(use_category, caption, category):
    caption = str(caption).strip().lower()
    category = str(category).strip().lower()
    return f"{category} {caption}" if use_category else caption

def svm_candidate_scores(bundle, caption, category):
    model_text = get_model_text(bundle["use_category"], caption, category)
    X = bundle["vectorizer"].transform([model_text])
    return bundle["model"].decision_function(X)[0]

def sbert_semantic_scores(caption, category, use_category, embedder, all_tags, hashtag_embeddings):
    model_text = get_model_text(use_category, caption, category)
    query_emb = embedder.encode([model_text], convert_to_numpy=True, normalize_embeddings=True)
    sims = cosine_similarity(query_emb, hashtag_embeddings)[0]
    return {tag: float(sim) for tag, sim in zip(all_tags, sims)}

def hybrid_rerank(bundle, exp, caption, category, trend_score_dict, category_affinity,
                  embedder, all_tags, hashtag_embeddings, candidate_pool=20, top_k=5):
    raw_scores = svm_candidate_scores(bundle, caption, category)
    candidates_idx = np.argsort(raw_scores)[::-1][:candidate_pool]
    cand_tags = [bundle["mlb"].classes_[i] for i in candidates_idx]

    base_dict = {tag: float(raw_scores[i]) for tag, i in zip(cand_tags, candidates_idx)}
    base_norm = minmax_normalize_dict(base_dict)

    sem_all = sbert_semantic_scores(
        caption, category, bundle["use_category"],
        embedder, all_tags, hashtag_embeddings
    )
    sem_dict = {tag: sem_all.get(tag, 0.0) for tag in cand_tags}
    sem_norm = minmax_normalize_dict(sem_dict)

    category = str(category).strip().lower()
    rows = []
    for tag in cand_tags:
        base_score = base_norm.get(tag, 0.0)
        sem_score = sem_norm.get(tag, 0.0)
        trend_score = float(trend_score_dict.get(tag, 0.0))
        cat_score = category_affinity.get(category, {}).get(tag, 0.0)
        lex_score = lexical_similarity(caption, tag)
        penalty = GENERIC_PENALTY if tag in GENERIC_TAGS else 0.0

        final_score = (
            W_BASE * base_score +
            W_SEM * sem_score +
            W_TREND * trend_score +
            W_CAT * cat_score +
            W_LEX * lex_score -
            penalty
        )

        rows.append({
            "tag": tag,
            "final_score": final_score,
            "base_score": base_score,
            "sem_score": sem_score,
            "trend_score": trend_score,
            "cat_score": cat_score,
            "lex_score": lex_score,
            "penalty": penalty
        })

    rows = sorted(rows, key=lambda x: x["final_score"], reverse=True)
    return [r["tag"] for r in rows[:top_k]], rows

# =========================================================
# STREAMLIT APP
# =========================================================
st.title("Hybrid Hashtag Recommendation Demo")
st.caption("IR-faithful deployment: SVM candidate generation + SBERT semantic reranking + trend-aware scoring")

df = load_dataset(str(DATA_PATH))
exp = prepare_experiment_data(df)
svm_bundle = load_svm_bundle(exp)
sbert_lr_bundle = load_sbert_lr_bundle()

category_affinity = build_category_affinity(exp["train_df"])
_, trend_score_dict = build_trend_score_table(exp["train_df"])
embedder_sem, all_tags_sem, hashtag_embeddings_sem = build_sbert_semantic_index(exp["train_df"])

categories = sorted(exp["train_df"]["category"].dropna().unique().tolist())

category = st.selectbox("Category", categories)
caption = st.text_area("Caption", "Having a great morning workout at the gym, feeling strong today!")
top_k = st.slider("Top K", 3, 10, 5)
candidate_pool = st.slider("Candidate Pool", 10, 50, 20, step=5)

if st.button("Recommend"):
    tags, rows = hybrid_rerank(
        svm_bundle, exp, caption, category,
        trend_score_dict, category_affinity,
        embedder_sem, all_tags_sem, hashtag_embeddings_sem,
        candidate_pool=candidate_pool, top_k=top_k
    )

    st.subheader("Final Recommended Hashtags")
    st.code(" ".join(tags))

    st.subheader("Scoring Breakdown")
    st.dataframe(pd.DataFrame(rows[:top_k]), use_container_width=True, hide_index=True)

    if sbert_lr_bundle is not None:
        st.subheader("SBERT + LR Comparison")
        model_text = get_model_text(sbert_lr_bundle["use_category"], caption, category)
        X = sbert_lr_bundle["vectorizer"].encode(
            [model_text],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        sbert_scores = sbert_lr_bundle["model"].predict_proba(X)[0]
        top_idx = np.argsort(sbert_scores)[::-1][:top_k]
        sbert_tags = [sbert_lr_bundle["mlb"].classes_[i] for i in top_idx]
        st.write(sbert_tags)
