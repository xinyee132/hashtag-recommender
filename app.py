import re
import ast
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="FYP Hashtag Recommender Demo", layout="wide")

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

# Final deployed trend-aware settings
TREND_W_BASE = 0.55
TREND_W_TREND = 0.20
TREND_W_CAT = 0.15
TREND_W_LEX = 0.10
TREND_GENERIC_PENALTY = 0.12

# Experimental hybrid settings
HYBRID_W_BASE = 0.40
HYBRID_W_TREND = 0.20
HYBRID_W_CAT = 0.10
HYBRID_W_LEX = 0.10
HYBRID_W_SEM = 0.20
HYBRID_GENERIC_PENALTY = 0.12

GENERIC_TAGS = {
    "#ad", "#love", "#instagood", "#photooftheday", "#photography",
    "#happy", "#happiness", "#life", "#beautiful", "#art", "#support"
}

SAMPLE_CAPTIONS = [
    ("Having a great morning workout at the gym, feeling strong today!", "fitness"),
    ("Enjoying a delicious homemade vegan pasta for dinner", "food"),
    ("Sunset view at the beach during vacation", "travel"),
    ("New outfit for the weekend brunch", "fashion"),
]

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
# DATA LOADING
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

# =========================================================
# EXPERIMENT PREP
# =========================================================
def prepare_experiment_data(df: pd.DataFrame, use_category=True, top_n_hashtags=TOP_N_HASHTAGS):
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

    categories = work_df["category"].values

    from sklearn.model_selection import train_test_split

    train_df, temp_df, y_train, y_temp = train_test_split(
        work_df,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=categories
    )

    temp_categories = temp_df["category"].values
    val_df, test_df, y_val, y_test = train_test_split(
        temp_df,
        y_temp,
        test_size=0.5,
        random_state=SEED,
        stratify=temp_categories
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
# MODEL TRAINING
# =========================================================
def run_caption_category_svm(exp):
    tfidf = TfidfVectorizer(
        max_features=MAX_FEATURES_TFIDF,
        ngram_range=(1, 1),
        sublinear_tf=True,
        min_df=MIN_DF,
        max_df=MAX_DF,
    )

    X_train = tfidf.fit_transform(exp["train_df"]["model_text"])
    X_val = tfidf.transform(exp["val_df"]["model_text"])
    X_test = tfidf.transform(exp["test_df"]["model_text"])

    base = LinearSVC(C=SVM_C, dual=False, max_iter=3000, random_state=SEED)
    model = OneVsRestClassifier(base)
    model.fit(X_train, exp["y_train"])

    return {
        "name": "caption_category_svm",
        "model": model,
        "vectorizer": tfidf,
        "mlb": exp["mlb"],
        "use_category": True,
        "model_type": "sklearn",
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test
    }

# =========================================================
# AUXILIARY TABLES
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

def build_hashtag_texts(train_df):
    tag_texts = defaultdict(list)

    for _, row in train_df.iterrows():
        caption = str(row["clean_caption"]).strip().lower()
        for tag in row["hashtags_list"]:
            tag_texts[tag].append(caption)

    return {tag: " ".join(texts) for tag, texts in tag_texts.items()}

def build_semantic_index(tfidf, train_df):
    tag_docs = build_hashtag_texts(train_df)
    tags = list(tag_docs.keys())
    docs = list(tag_docs.values())
    tag_vectors = tfidf.transform(docs)

    return {
        "tags": tags,
        "tag_docs": tag_docs,
        "tag_vectors": tag_vectors
    }

# =========================================================
# BUILD SYSTEM
# =========================================================
@st.cache_resource
def build_system(csv_path: str):
    df = load_dataset(csv_path)
    exp = prepare_experiment_data(df, use_category=True, top_n_hashtags=TOP_N_HASHTAGS)
    bundle = run_caption_category_svm(exp)

    category_affinity = build_category_affinity(exp["train_df"])
    trend_df, trend_score_dict = build_trend_score_table(exp["train_df"])
    semantic_index = build_semantic_index(bundle["vectorizer"], exp["train_df"])

    return {
        "df": df,
        "exp": exp,
        "bundle": bundle,
        "category_affinity": category_affinity,
        "trend_df": trend_df,
        "trend_score_dict": trend_score_dict,
        "semantic_index": semantic_index
    }

# =========================================================
# INFERENCE
# =========================================================
def raw_predict(system, caption: str, category: str, top_k=5, candidate_pool=20):
    bundle = system["bundle"]
    category = str(category).strip().lower()
    caption = str(caption).strip().lower()
    model_text = f"{category} {caption}"

    X = bundle["vectorizer"].transform([model_text])
    scores = bundle["model"].decision_function(X)[0]

    top_idx = np.argsort(scores)[::-1][:top_k]
    tags = [bundle["mlb"].classes_[i] for i in top_idx]

    cand_idx = np.argsort(scores)[::-1][:candidate_pool]
    cand_tags = [bundle["mlb"].classes_[i] for i in cand_idx]
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
        penalty = TREND_GENERIC_PENALTY if tag in GENERIC_TAGS else 0.0

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
        penalty = TREND_GENERIC_PENALTY if tag in GENERIC_TAGS else 0.0

        final_score = (
            TREND_W_BASE * base_score +
            TREND_W_TREND * trend_score +
            TREND_W_CAT * cat_score +
            TREND_W_LEX * lex_score -
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

def compute_semantic_scores(system, caption: str, category: str):
    bundle = system["bundle"]
    category = str(category).strip().lower()
    caption = str(caption).strip().lower()
    model_text = f"{category} {caption}"

    query_vec = bundle["vectorizer"].transform([model_text])
    sims = cosine_similarity(query_vec, system["semantic_index"]["tag_vectors"])[0]

    return {tag: float(sim) for tag, sim in zip(system["semantic_index"]["tags"], sims)}

def hybrid_rerank(system, caption: str, category: str, candidate_pool=20, top_k=5):
    category = str(category).strip().lower()
    caption = str(caption).strip().lower()

    _, cand_scores = raw_predict(system, caption, category, top_k=top_k, candidate_pool=candidate_pool)
    cand_norm = minmax_normalize_dict(cand_scores)

    semantic_scores_all = compute_semantic_scores(system, caption, category)
    cand_sem = {tag: semantic_scores_all.get(tag, 0.0) for tag in cand_scores.keys()}
    cand_sem_norm = minmax_normalize_dict(cand_sem)

    rows = []
    for tag in cand_scores.keys():
        base_score = cand_norm.get(tag, 0.0)
        trend_score = float(system["trend_score_dict"].get(tag, 0.0))
        cat_score = system["category_affinity"].get(category, {}).get(tag, 0.0)
        lex_score = lexical_similarity(caption, tag)
        sem_score = cand_sem_norm.get(tag, 0.0)
        penalty = HYBRID_GENERIC_PENALTY if tag in GENERIC_TAGS else 0.0

        final_score = (
            HYBRID_W_BASE * base_score +
            HYBRID_W_TREND * trend_score +
            HYBRID_W_CAT * cat_score +
            HYBRID_W_LEX * lex_score +
            HYBRID_W_SEM * sem_score -
            penalty
        )

        rows.append({
            "tag": tag,
            "final_score": float(final_score),
            "base_score": float(base_score),
            "trend_score": float(trend_score),
            "cat_score": float(cat_score),
            "lex_score": float(lex_score),
            "sem_score": float(sem_score),
            "penalty": float(penalty)
        })

    rows = sorted(rows, key=lambda x: x["final_score"], reverse=True)
    return [r["tag"] for r in rows[:top_k]], rows

def get_top_category_trends(system, category: str, top_n=10):
    train_df = system["exp"]["train_df"]
    group = train_df[train_df["category"] == category].copy()
    if len(group) == 0:
        return pd.DataFrame()

    trend_df_cat, _ = build_trend_score_table(group, recent_days=RECENT_DAYS, older_days=OLDER_DAYS, min_freq=2)
    return trend_df_cat.head(top_n)

# =========================================================
# EVALUATION HELPERS
# =========================================================
def evaluate_rank_metrics(y_true, y_scores, name="MODEL", k_values=[1, 3, 5]):
    from sklearn.metrics import ndcg_score, label_ranking_average_precision_score

    results = {}
    lrap = label_ranking_average_precision_score(y_true, y_scores)
    results["LRAP"] = float(lrap)

    n_samples = y_true.shape[0]
    valid_samples = 0

    metrics = {
        "P": {k: 0.0 for k in k_values},
        "R": {k: 0.0 for k in k_values},
        "Hit": {k: 0.0 for k in k_values},
        "MAP": {k: 0.0 for k in k_values},
    }

    for i in range(n_samples):
        true_indices = np.where(y_true[i] == 1)[0]
        if len(true_indices) == 0:
            continue

        valid_samples += 1
        ranked_indices = np.argsort(y_scores[i])[::-1]

        for k in k_values:
            top_k = ranked_indices[:k]
            hits = len(set(top_k).intersection(set(true_indices)))

            metrics["P"][k] += hits / k
            metrics["R"][k] += hits / len(true_indices)
            metrics["Hit"][k] += 1 if hits > 0 else 0

            ap = 0.0
            hits_so_far = 0
            for rank, pred_idx in enumerate(top_k, start=1):
                if pred_idx in true_indices:
                    hits_so_far += 1
                    ap += hits_so_far / rank
            metrics["MAP"][k] += ap / min(k, len(true_indices))

    for k in k_values:
        p_at_k = metrics["P"][k] / valid_samples
        r_at_k = metrics["R"][k] / valid_samples
        f1_at_k = 2 * (p_at_k * r_at_k) / (p_at_k + r_at_k) if (p_at_k + r_at_k) > 0 else 0.0
        hit_at_k = metrics["Hit"][k] / valid_samples
        map_at_k = metrics["MAP"][k] / valid_samples
        ndcg_at_k = ndcg_score(y_true, y_scores, k=k)

        results[f"P@{k}"] = float(p_at_k)
        results[f"R@{k}"] = float(r_at_k)
        results[f"F1@{k}"] = float(f1_at_k)
        results[f"Hit@{k}"] = float(hit_at_k)
        results[f"MAP@{k}"] = float(map_at_k)
        results[f"NDCG@{k}"] = float(ndcg_at_k)

    return results

def tags_to_score_vector(tags, mlb, full_length_score=False):
    vec = np.zeros(len(mlb.classes_))
    tag_to_idx = {t: i for i, t in enumerate(mlb.classes_)}

    n = len(tags)
    for rank, tag in enumerate(tags):
        if tag in tag_to_idx:
            vec[tag_to_idx[tag]] = (n - rank) if full_length_score else 1.0

    return vec

def get_raw_scores(system, split="val"):
    bundle = system["bundle"]
    exp = system["exp"]
    df = exp[f"{split}_df"]
    texts = df["model_text"].tolist()

    X = bundle["vectorizer"].transform(texts)
    return bundle["model"].decision_function(X)

def get_lexical_scores(system, split="val", top_k=5):
    df = system["exp"][f"{split}_df"]
    all_scores = []

    for _, row in df.iterrows():
        tags, _ = lexical_rerank(system, row["clean_caption"], row["category"], top_k=top_k)
        vec = tags_to_score_vector(tags, system["bundle"]["mlb"], full_length_score=True)
        all_scores.append(vec)

    return np.array(all_scores)

def get_trend_scores(system, split="val", top_k=5):
    df = system["exp"][f"{split}_df"]
    all_scores = []

    for _, row in df.iterrows():
        tags, _ = trend_aware_rerank(system, row["clean_caption"], row["category"], top_k=top_k)
        vec = tags_to_score_vector(tags, system["bundle"]["mlb"], full_length_score=True)
        all_scores.append(vec)

    return np.array(all_scores)

def get_hybrid_scores(system, split="val", top_k=5):
    df = system["exp"][f"{split}_df"]
    all_scores = []

    for _, row in df.iterrows():
        tags, _ = hybrid_rerank(system, row["clean_caption"], row["category"], top_k=top_k)
        vec = tags_to_score_vector(tags, system["bundle"]["mlb"], full_length_score=True)
        all_scores.append(vec)

    return np.array(all_scores)

def summarize_results(results_dict, split_name="val"):
    rows = []
    for method, metrics in results_dict.items():
        rows.append({
            "split": split_name,
            "method": method,
            "LRAP": metrics["LRAP"],
            "P@1": metrics["P@1"],
            "P@3": metrics["P@3"],
            "P@5": metrics["P@5"],
            "R@3": metrics["R@3"],
            "R@5": metrics["R@5"],
            "F1@3": metrics["F1@3"],
            "F1@5": metrics["F1@5"],
            "Hit@3": metrics["Hit@3"],
            "Hit@5": metrics["Hit@5"],
            "MAP@5": metrics["MAP@5"],
            "NDCG@5": metrics["NDCG@5"],
        })
    return pd.DataFrame(rows)

def run_demo_evaluation(system):
    exp = system["exp"]

    val_results = {
        "raw": evaluate_rank_metrics(exp["y_val"], get_raw_scores(system, "val"), "RAW"),
        "lexical": evaluate_rank_metrics(exp["y_val"], get_lexical_scores(system, "val"), "LEXICAL"),
        "trend": evaluate_rank_metrics(exp["y_val"], get_trend_scores(system, "val"), "TREND"),
        "hybrid": evaluate_rank_metrics(exp["y_val"], get_hybrid_scores(system, "val"), "HYBRID"),
    }

    test_results = {
        "raw": evaluate_rank_metrics(exp["y_test"], get_raw_scores(system, "test"), "RAW"),
        "lexical": evaluate_rank_metrics(exp["y_test"], get_lexical_scores(system, "test"), "LEXICAL"),
        "trend": evaluate_rank_metrics(exp["y_test"], get_trend_scores(system, "test"), "TREND"),
        "hybrid": evaluate_rank_metrics(exp["y_test"], get_hybrid_scores(system, "test"), "HYBRID"),
    }

    val_df = summarize_results(val_results, "val")
    test_df = summarize_results(test_results, "test")
    return pd.concat([val_df, test_df], ignore_index=True)

# =========================================================
# UI
# =========================================================
st.title("FYP Hybrid Hashtag Recommender Demo")
st.caption("Caption + Category SVM with lexical, trend-aware, and TF-IDF semantic hybrid reranking.")

system = build_system(str(DATA_PATH))
categories = sorted(system["exp"]["train_df"]["category"].dropna().unique().tolist())

tab1, tab2, tab3 = st.tabs(["Single Prediction", "Qualitative Demo", "Evaluation Demo"])

with tab1:
    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Input")
        default_idx = categories.index("fitness") if "fitness" in categories else 0
        category = st.selectbox("Category", categories, index=default_idx)

        caption = st.text_area(
            "Caption",
            value="Having a great morning workout at the gym, feeling strong today!",
            height=140
        )

        top_k = st.slider("Top hashtags", min_value=3, max_value=10, value=5)
        candidate_pool = st.slider("Candidate pool", min_value=10, max_value=50, value=20, step=5)

        run_btn = st.button("Recommend Hashtags", type="primary")

    with right:
        st.subheader("Current Category Trends")
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
            raw_tags, _ = raw_predict(system, caption, category, top_k=top_k, candidate_pool=candidate_pool)
            lex_tags, lex_rows = lexical_rerank(system, caption, category, candidate_pool=candidate_pool, top_k=top_k)
            trend_tags, trend_rows = trend_aware_rerank(system, caption, category, candidate_pool=candidate_pool, top_k=top_k)
            hybrid_tags, hybrid_rows = hybrid_rerank(system, caption, category, candidate_pool=candidate_pool, top_k=top_k)

            st.subheader("Recommendation Output")
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.markdown("**Raw SVM**")
                for i, tag in enumerate(raw_tags, 1):
                    st.write(f"{i}. {tag}")

            with c2:
                st.markdown("**Lexical Rerank**")
                for i, tag in enumerate(lex_tags, 1):
                    st.write(f"{i}. {tag}")

            with c3:
                st.markdown("**Trend-Aware**")
                for i, tag in enumerate(trend_tags, 1):
                    st.write(f"{i}. {tag}")

            with c4:
                st.markdown("**Hybrid**")
                for i, tag in enumerate(hybrid_tags, 1):
                    st.write(f"{i}. {tag}")

            st.subheader("Final Hashtags (Hybrid Demo)")
            st.code(" ".join(hybrid_tags))

            st.subheader("Hybrid Scoring Breakdown")
            hybrid_breakdown_df = pd.DataFrame(hybrid_rows[:top_k])
            st.dataframe(
                hybrid_breakdown_df[[
                    "tag", "final_score", "base_score", "trend_score",
                    "cat_score", "lex_score", "sem_score", "penalty"
                ]],
                use_container_width=True,
                hide_index=True
            )

with tab2:
    st.subheader("Qualitative Comparison")
    selected_sample = st.selectbox(
        "Choose demo sample",
        options=list(range(len(SAMPLE_CAPTIONS))),
        format_func=lambda i: f"{SAMPLE_CAPTIONS[i][1]} — {SAMPLE_CAPTIONS[i][0][:50]}..."
    )

    sample_caption, sample_category = SAMPLE_CAPTIONS[selected_sample]

    raw_tags, _ = raw_predict(system, sample_caption, sample_category, top_k=5, candidate_pool=20)
    lex_tags, _ = lexical_rerank(system, sample_caption, sample_category, candidate_pool=20, top_k=5)
    trend_tags, _ = trend_aware_rerank(system, sample_caption, sample_category, candidate_pool=20, top_k=5)
    hybrid_tags, _ = hybrid_rerank(system, sample_caption, sample_category, candidate_pool=20, top_k=5)

    st.markdown(f"**Caption:** {sample_caption}")
    st.markdown(f"**Category:** {sample_category}")

    compare_df = pd.DataFrame({
        "Rank": [1, 2, 3, 4, 5],
        "RAW": raw_tags,
        "LEXICAL": lex_tags,
        "TREND": trend_tags,
        "HYBRID": hybrid_tags
    })
    st.dataframe(compare_df, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Evaluation Demo")
    st.caption("This may take some time because it runs reranking across validation and test data.")

    if st.button("Run Evaluation Table"):
        with st.spinner("Running evaluation..."):
            results_df = run_demo_evaluation(system)

        st.dataframe(results_df, use_container_width=True, hide_index=True)

        st.markdown("**Suggested interpretation**")
        st.write(
            "Use this table in your FYP demo to show that trend-aware reranking improves top-K usefulness, "
            "while the hybrid semantic model can be discussed as an experimental extension."
        )
