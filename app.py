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
from sklearn.preprocessing import MultiLabelBinarizer

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Trend-Aware Hashtag Intelligence", layout="wide")

# =========================================================
# CONSTANTS & CONFIG (Synced exactly with Thesis & Eval Script)
# =========================================================
RERANK_W_BASE   = 0.50  # Semantic/Model backbone (SBERT)
RERANK_W_TREND  = 0.20  # Engagement Intensity
RERANK_W_CAT    = 0.15  # Historical Category Affinity
RERANK_W_LEX    = 0.15  # Direct Caption Keyword Match
GENERIC_PENALTY = 0.12  # Penalty for non-informative tags

GENERIC_TAGS = {"#ad", "#love", "#instagood", "#photooftheday", "#happy", "#life", "#beautiful"}
MODEL_DIR = Path("saved_models")
DATA_PATH = Path("instagram_dataset_tfidf_ready.csv")

# =========================================================
# CORE FUNCTIONS
# =========================================================
def safe_parse(x):
    try:
        return ast.literal_eval(x) if isinstance(x, str) else []
    except: return []

def tokenize(text):
    return re.sub(r"[^a-z0-9\s]", " ", str(text).lower()).split()

def hashtag_token(tag):
    return re.sub(r"[^a-z0-9]", "", tag.lower().replace("#", "")).strip()

def minmax_normalize_dict(d):
    if not d: return {}
    vals = np.array(list(d.values()), dtype=float)
    if vals.max() == vals.min(): return {k: 0.0 for k in d}
    return {k: (v - vals.min()) / (vals.max() - vals.min()) for k, v in d.items()}

def calculate_lexical_sim(caption, tag):
    words = set(tokenize(caption))
    token = hashtag_token(tag)
    if token in words: return 1.0
    for w in words:
        if token in w or w in token: return 0.6
    return 0.0

# =========================================================
# ACTUAL MODEL LOADER: SBERT + LR
# =========================================================
@st.cache_resource
def load_sbert_lr_bundle(export_dir=MODEL_DIR):
    export_dir = Path(export_dir)
    clf_path = export_dir / "sbert_lr_clf.joblib"
    meta_path = export_dir / "sbert_lr_meta.pkl"

    if not clf_path.exists() or not meta_path.exists():
        st.error(f"Model files not found in {export_dir}. Please ensure joblib and pkl files exist.")
        st.stop()

    clf = joblib.load(clf_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    embedder = SentenceTransformer(meta["model_name"])

    return {
        "model": clf,
        "vectorizer": embedder,
        "mlb": meta["mlb"]
    }

# =========================================================
# DATA & TREND DETECTION
# =========================================================
@st.cache_data
def load_and_process_data(csv_path):
    df = pd.read_csv(csv_path)
    df["hashtags_list"] = df["hashtags_list"].apply(safe_parse)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    
    # Engagement Rate Formula
    if "followers" not in df.columns: df["followers"] = 1000 
    df["engagement_rate"] = (df["likes"] + df["comment_count"]) / (df["followers"] + 1)
    
    return df

def build_trend_intelligence(df, days=30):
    max_date = df["timestamp"].max()
    recent_cutoff = max_date - pd.Timedelta(days=days)
    
    recent_df = df[df["timestamp"] >= recent_cutoff]
    older_df = df[df["timestamp"] < recent_cutoff]
    
    stats = []
    all_tags = set([t for sub in df["hashtags_list"] for t in sub])
    
    for tag in all_tags:
        r_count = sum(recent_df["hashtags_list"].apply(lambda x: tag in x))
        o_count = sum(older_df["hashtags_list"].apply(lambda x: tag in x))
        
        r_eng = recent_df[recent_df["hashtags_list"].apply(lambda x: tag in x)]["engagement_rate"].mean()
        o_eng = older_df[older_df["hashtags_list"].apply(lambda x: tag in x)]["engagement_rate"].mean()
        
        if pd.isna(r_eng): r_eng = 0
        if pd.isna(o_eng): o_eng = 0
        if r_count + o_count < 5: continue
            
        velocity = (r_count + 1) / (o_count + 1)
        eng_growth = (r_eng + 0.01) / (o_eng + 0.01)
        trend_score = (velocity * 0.6) + (eng_growth * 0.4)
        
        stats.append({
            "Hashtag": tag,
            "Velocity": velocity,
            "Engagement Growth": eng_growth,
            "Trend Score": trend_score
        })
    
    trend_df = pd.DataFrame(stats).sort_values("Trend Score", ascending=False)
    max_trend = trend_df["Trend Score"].max()
    if pd.isna(max_trend) or max_trend == 0:
        trend_dict = {row["Hashtag"]: 0.0 for _, row in trend_df.iterrows()}
    else:
        trend_dict = {row["Hashtag"]: row["Trend Score"]/max_trend for _, row in trend_df.iterrows()}
    
    return trend_df, trend_dict

def build_category_affinity(df):
    cat_tag_counts = defaultdict(Counter)
    for _, row in df.iterrows():
        cat = str(row["category"]).lower().strip()
        for tag in row["hashtags_list"]:
            cat_tag_counts[cat][tag] += 1
    
    affinity = {}
    for cat, counter in cat_tag_counts.items():
        total = sum(counter.values())
        affinity[cat] = {tag: count/total for tag, count in counter.items()}
    return affinity

# =========================================================
# INITIALIZATION
# =========================================================
try:
    df = load_and_process_data(DATA_PATH)
    trend_df, trend_dict = build_trend_intelligence(df)
    cat_affinity = build_category_affinity(df)
    bundle = load_sbert_lr_bundle(MODEL_DIR)
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

# =========================================================
# UI LAYOUT
# =========================================================
st.title("📱 Social Media Trend & Hashtag Intelligence")

tab1, tab2 = st.tabs(["🔍 Smart Recommendation", "📈 Global Trend Dashboard"])

with tab1:
    col_in, col_out = st.columns([1, 1])
    
    with col_in:
        st.subheader("Post Details")
        category = st.selectbox("Category", sorted(df["category"].dropna().unique()))
        caption = st.text_area("Caption", placeholder="Enter your Instagram caption here...", height=120)
        top_k = st.slider("Result Count", 3, 10, 5)
        
        st.subheader("System Architecture")
        st.info("Active Backbone: SBERT + Logistic Regression")
        st.caption("Reranker Weights: 50% Semantic (SBERT), 20% Trend, 15% Category, 15% Lexical")

    with col_out:
        st.subheader("Recommendations")
        if st.button("Generate Optimized Tags", type="primary"):
            if not caption.strip():
                st.warning("Please provide a caption.")
            else:
                with st.spinner("Generating semantic embeddings..."):
                    # 1. ACTUAL SBERT INFERENCE
                    model_text = f"{str(category).lower()} {str(caption).lower()}"
                    X_emb = bundle["vectorizer"].encode([model_text], convert_to_numpy=True, normalize_embeddings=True)
                    
                    if hasattr(bundle["model"], "predict_proba"):
                        raw_scores = bundle["model"].predict_proba(X_emb)[0]
                    else:
                        raw_scores = bundle["model"].decision_function(X_emb)[0]

                    # 2. CANDIDATE GENERATION (Top 20 from SBERT)
                    candidate_pool = 20
                    top_indices = np.argsort(raw_scores)[::-1][:candidate_pool]
                    candidate_tags = [bundle["mlb"].classes_[i] for i in top_indices]
                    
                    base_scores = {tag: float(raw_scores[i]) for tag, i in zip(candidate_tags, top_indices)}
                    base_norm = minmax_normalize_dict(base_scores)
                    
                    # 3. HYBRID RERANKING
                    results = []
                    for tag in candidate_tags:
                        s_base = base_norm.get(tag, 0.0)
                        s_trend = trend_dict.get(tag, 0.0)
                        s_cat = cat_affinity.get(category.lower(), {}).get(tag, 0.0)
                        s_lex = calculate_lexical_sim(caption, tag)
                        penalty = GENERIC_PENALTY if tag in GENERIC_TAGS else 0.0
                        
                        final_score = (RERANK_W_BASE * s_base) + \
                                      (RERANK_W_TREND * s_trend) + \
                                      (RERANK_W_CAT * s_cat) + \
                                      (RERANK_W_LEX * s_lex) - penalty
                        
                        results.append({
                            "Hashtag": tag,
                            "Score": final_score,
                            "SBERT": s_base,
                            "Lexical": s_lex,
                            "Trend": s_trend,
                            "Category": s_cat
                        })
                    
                    final_df = pd.DataFrame(results).sort_values("Score", ascending=False).head(top_k)
                    
                    st.success("Targeted Hashtags Generated:")
                    st.markdown(f"### `{' '.join(final_df['Hashtag'].tolist())}`")
                    
                    st.write("---")
                    st.subheader("Explainability Breakdown")
                    # Formatted explainability string
                    final_df["Why?"] = final_df.apply(lambda x: 
                        f"Semantics: {int(x['SBERT']*100)}% | Match: {int(x['Lexical']*100)}% | Trend: {int(x['Trend']*100)}% | Cat: {int(x['Category']*100)}%", axis=1)
                    st.table(final_df[["Hashtag", "Why?"]])

with tab2:
    st.subheader("Global Trend Intelligence")
    st.caption("Calculated using Engagement Rate = (Likes + Comments) / (Followers + 1)")
    
    if len(trend_df) > 0:
        m1, m2, m3 = st.columns(3)
        m1.metric("Viral Potential", trend_df.iloc[0]["Hashtag"], f"{trend_df.iloc[0]['Velocity']:.1f}x Velocity")
        m2.metric("Engagement Leader", trend_df.sort_values("Engagement Growth").iloc[-1]["Hashtag"], "High Interaction")
        m3.metric("System Safety", "Active", f"{len(GENERIC_TAGS)} Generic Tags Penalized")
        
        st.write("---")
        st.write("**Explainable Trend Data**")
        
        display_trends = trend_df.head(15).copy()
        display_trends["Strength"] = display_trends["Trend Score"].apply(
            lambda x: "🔥 High" if x > 0.7 else ("⚡ Medium" if x > 0.4 else "✅ Stable"))
        
        st.dataframe(display_trends[[
            "Hashtag", "Strength", "Velocity", "Engagement Growth", "Trend Score"
        ]], use_container_width=True, hide_index=True)
        
        st.caption("Velocity: Usage Frequency Shift | Engagement Growth: Interaction Intensity Shift")
    else:
        st.info("Not enough temporal data to calculate trends.")
