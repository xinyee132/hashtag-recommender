import re
import ast
import pickle
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer

# =========================================================
# CONFIG
# =========================================================
BASE_DIR = Path(".")
DATA_PATH = BASE_DIR / "instagram_dataset_tfidf_ready.csv"

TOP_N_HASHTAGS = 400
MAX_FEATURES_TFIDF = 2000
MIN_DF = 10
MAX_DF = 0.85
SVM_C = 0.05
SEED = 42

RECENT_DAYS = 30
OLDER_DAYS = 90
TREND_MIN_FREQ = 3

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

def tokenize(text: str):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()

def hashtag_token(tag: str):
    tag = str(tag).lower().replace("#", "")
    tag = re.sub(r"[_\-]", " ", tag)
    tag = re.sub(r"[^a-z0-9\s]", "", tag)
    tag = re.sub(r"\s+", " ", tag).strip()
    return tag

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

def lexical_similarity(caption, tag):
    words = set(tokenize(caption))
    token = hashtag_token(tag)

    if token in words:
        return 1.0

    for w in words:
        if token in w or w in token:
            return 0.6

    return 0.0

# =========================================================
# DATA + TRAINING
# =========================================================
@st.cache_data
def load_dataset(csv_path: str):
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

def prepare_deploy_data(df: pd.DataFrame, top_n_hashtags=TOP_N_HASHTAGS):
    all_tags = [tag for tags in df["hashtags_list"] for tag in tags]
    top_tags = set([tag for tag, _ in Counter(all_tags).most_common(top_n_hashtags)])

    work_df = df.copy()
    work_df["hashtags_list"] = work_df["hashtags_list"].apply(lambda tags: [t for t in tags if t in top_tags])
    work_df = work_df[work_df["hashtags_list"].map(len) > 0].reset_index(drop=True)

    work_df["model_text"] = work_df["category"] + " " + work_df["clean_caption"]

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(work_df["hashtags_list"])

    tfidf = TfidfVectorizer(
        max_features=MAX_FEATURES_TFIDF,
        ngram_range=(1, 1),
        sublinear_tf=True,
        min_df=MIN_DF,
        max_df=MAX_DF,
    )
    X = tfidf.fit_transform(work_df["model_text"])

    base = LinearSVC(C=SVM_C, dual=False, max_iter=3000, random_state=SEED)
    model = OneVsRestClassifier(base)
    model.fit(X, y)

    return work_df, mlb, tfidf, model

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
        tags = row["hashtags_list"]
        eng = float(row["engagement"])
        age_days = max((max_time - row["timestamp"]).days, 0)
        recency_weight = 1 / (1 + age_days)

        for tag in tags:
            recent_counts[tag] += 1
            recent_eng_sum[tag] += eng
            recent_recency_sum[tag] += recency_weight

    for _, row in older_df.iterrows():
        tags = row["hashtags_list"]
        eng = float(row["engagement"])

        for tag in tags:
            older_counts[tag] += 1
            older_eng_sum[tag] += eng

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

        rows.append({
            "tag": tag,
            "recent_count": r_count,
            "older_count": o_count,
            "total_freq": total_freq,
            "recent_avg_engagement": r_avg_eng,
            "older_avg_engagement": o_avg_eng,
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

    trend_df = trend_df.sort_values("trend_score", ascending=False).reset_index(drop=True)
    trend_score_dict = dict(zip(trend_df["tag"], trend_df["trend_score"]))
    return trend_df, trend_score_dict

@st.cache_resource
def build_system(csv_path: str):
    df = load_dataset(csv_path)
    train_df, mlb, tfidf, model = prepare_deploy_data(df)
    category_affinity = build_category_affinity(train_df)
    trend_df, trend_score_dict = build_trend_score_table(train_df)

    return {
        "df": train_df,
        "mlb": mlb,
        "tfidf": tfidf,
        "model": model,
        "category_affinity": category_affinity,
        "trend_df": trend_df,
        "trend_score_dict": trend_score_dict,
    }

# =========================================================
# INFERENCE
# =========================================================
def raw_predict(system, caption: str, category: str, top_k=5, candidate_pool=20):
    category = str(category).strip().lower()
    caption = str(caption).strip().lower()
    model_text = f"{category} {caption}"

    X = system["tfidf"].transform([model_text])
    scores = system["model"].decision_function(X)[0]

    top_idx = np.argsort(scores)[::-1][:top_k]
    tags = [system["mlb"].classes_[i] for i in top_idx]

    cand_idx = np.argsort(scores)[::-1][:candidate_pool]
    cand_tags = [system["mlb"].classes_[i] for i in cand_idx]
    cand_scores = {tag: float(scores[i]) for tag, i in zip(cand_tags, cand_idx)}

    return tags, cand_scores

def lexical_rerank(system, caption: str, category: str, candidate_pool=20, top_k=5):
    category = str(category).strip().lower()
    caption = str(caption).strip().lower()

    _, cand_scores = raw_predict(system, caption, category, top_k=top_k, candidate_pool=candidate_pool)
    cand_norm = minmax_normalize_dict(cand_scores)

    rows = []
    for tag in cand_scores.keys():
        base_score = cand_norm.get(tag, 0.0)
        cat_score = system["category_affinity"].get(category, {}).get(tag, 0.0)
        lex_score = lexical_similarity(caption, tag)
        penalty = 0.12 if tag in GENERIC_TAGS else 0.0

        final_score = 0.75 * base_score + 0.15 * cat_score + 0.10 * lex_score - penalty
        rows.append({
            "tag": tag,
            "final_score": float(final_score),
            "base_score": float(base_score),
            "cat_score": float(cat_score),
            "lex_score": float(lex_score),
            "penalty": float(penalty)
        })

    rows = sorted(rows, key=lambda x: x["final_score"], reverse=True)
    return [r["tag"] for r in rows[:top_k]], rows

def trend_aware_rerank(system, caption: str, category: str, candidate_pool=20, top_k=5):
    category = str(category).strip().lower()
    caption = str(caption).strip().lower()

    _, cand_scores = raw_predict(system, caption, category, top_k=top_k, candidate_pool=candidate_pool)
    cand_norm = minmax_normalize_dict(cand_scores)

    rows = []
    for tag in cand_scores.keys():
        base_score = cand_norm.get(tag, 0.0)
        trend_score = float(system["trend_score_dict"].get(tag, 0.0))
        cat_score = system["category_affinity"].get(category, {}).get(tag, 0.0)
        lex_score = lexical_similarity(caption, tag)
        penalty = 0.12 if tag in GENERIC_TAGS else 0.0

        final_score = (
            0.55 * base_score +
            0.20 * trend_score +
            0.15 * cat_score +
            0.10 * lex_score -
            penalty
        )

        rows.append({
            "tag": tag,
            "final_score": float(final_score),
            "base_score": float(base_score),
            "trend_score": float(trend_score),
            "cat_score": float(cat_score),
            "lex_score": float(lex_score),
            "penalty": float(penalty)
        })

    rows = sorted(rows, key=lambda x: x["final_score"], reverse=True)
    return [r["tag"] for r in rows[:top_k]], rows

def get_top_category_trends(system, category: str, top_n=10):
    df = system["trend_df"]
    if len(df) == 0:
        return pd.DataFrame()

    # category trend table on the fly
    category_rows = []
    train_df = system["df"]
    group = train_df[train_df["category"] == category].copy()
    if len(group) == 0:
        return pd.DataFrame()

    trend_df_cat, _ = build_trend_score_table(group, recent_days=RECENT_DAYS, older_days=OLDER_DAYS, min_freq=2)
    return trend_df_cat.head(top_n)

# =========================================================
# STREAMLIT UI
# =========================================================
st.set_page_config(page_title="Trend-Aware Hashtag Recommender", layout="wide")
st.title("Trend-Aware Instagram Hashtag Recommender")
st.caption("Deployable pipeline: Caption + Category SVM with lexical and trend-aware reranking.")

system = build_system(str(DATA_PATH))

categories = sorted(system["df"]["category"].dropna().unique().tolist())

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Input")
    category = st.selectbox("Category", categories, index=categories.index("fitness") if "fitness" in categories else 0)
    caption = st.text_area(
        "Caption",
        value="Having a great morning workout at the gym, feeling strong today!",
        height=140
    )
    top_k = st.slider("Top hashtags", min_value=3, max_value=10, value=5)
    candidate_pool = st.slider("Candidate pool", min_value=10, max_value=50, value=20, step=5)

    run_btn = st.button("Recommend Hashtags", type="primary")

with right:
    st.subheader("Current Trends")
    top_trends = get_top_category_trends(system, category, top_n=10)
    if len(top_trends) > 0:
        st.dataframe(
            top_trends[["tag", "recent_count", "velocity", "engagement_growth", "trend_score"]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No category-specific trends available.")

if run_btn:
    if not caption.strip():
        st.warning("Please enter a caption.")
    else:
        raw_tags, raw_scores = raw_predict(system, caption, category, top_k=top_k, candidate_pool=candidate_pool)
        lex_tags, lex_rows = lexical_rerank(system, caption, category, candidate_pool=candidate_pool, top_k=top_k)
        trend_tags, trend_rows = trend_aware_rerank(system, caption, category, candidate_pool=candidate_pool, top_k=top_k)

        st.subheader("Recommendation Output")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Raw SVM**")
            for i, tag in enumerate(raw_tags, 1):
                st.write(f"{i}. {tag}")

        with c2:
            st.markdown("**Lexical Rerank**")
            for i, tag in enumerate(lex_tags, 1):
                st.write(f"{i}. {tag}")

        with c3:
            st.markdown("**Trend-Aware Final Output**")
            for i, tag in enumerate(trend_tags, 1):
                st.write(f"{i}. {tag}")

        st.subheader("Final Hashtags")
        st.code(" ".join(trend_tags))

        st.subheader("Trend-Aware Scoring Breakdown")
        trend_breakdown_df = pd.DataFrame(trend_rows[:top_k])
        st.dataframe(
            trend_breakdown_df[["tag", "final_score", "base_score", "trend_score", "cat_score", "lex_score", "penalty"]],
            use_container_width=True,
            hide_index=True
        )