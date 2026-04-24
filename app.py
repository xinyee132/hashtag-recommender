# =========================================================
# UI
# =========================================================
st.title("Hybrid Hashtag Recommendation Demo")

st.caption(
    "This demo compares three model families: "
    "Final Model 1 (SVM + TF-IDF semantic), "
    "Final Model 2 (SVM + BERT semantic), "
    "and SBERT + LR."
)

system = build_app_objects(str(DATA_PATH))
categories = sorted(system["exp"]["train_df"]["category"].dropna().unique().tolist())
family_options = get_family_options(system)

tab1, tab2, tab3 = st.tabs([
    "Single Prediction",
    "Qualitative Demo",
    "Evaluation Demo"
])

# =========================================================
# TAB 1: SINGLE PREDICTION
# =========================================================
with tab1:
    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Input")

        default_idx = categories.index("fitness") if "fitness" in categories else 0

        category = st.selectbox(
            "Category",
            categories,
            index=default_idx
        )

        caption = st.text_area(
            "Caption",
            value="Having a great morning workout at the gym, feeling strong today!",
            height=140
        )

        selected_family = st.selectbox(
            "Choose model family",
            family_options,
            index=0
        )

        top_k = st.slider(
            "Number of hashtags",
            min_value=3,
            max_value=10,
            value=5
        )

        # Candidate pool is fixed internally to reduce user complexity
        candidate_pool = 20

        run_btn = st.button(
            "Recommend Hashtags",
            type="primary"
        )

    with right:
        st.subheader("Current Overall Trends")

        trend_df = system["trend_df"]

        if len(trend_df) > 0:
            st.dataframe(
                trend_df[
                    ["tag", "recent_count", "velocity", "engagement_growth", "trend_score"]
                ].head(10),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No trend data available.")

    if run_btn:
        if not caption.strip():
            st.warning("Please enter a caption.")
        else:
            result = run_model_family(
                system=system,
                family_name=selected_family,
                caption=caption,
                category=category,
                top_k=top_k,
                candidate_pool=candidate_pool
            )

            st.divider()

            st.subheader("Recommended Hashtags")

            st.code(" ".join(result["final_tags"]))

            st.markdown("### Final Output")

            for i, tag in enumerate(result["final_tags"], 1):
                st.write(f"{i}. {tag}")

            with st.expander("Show model comparison details"):
                cols = st.columns(4)

                with cols[0]:
                    st.markdown("**Raw**")
                    for i, tag in enumerate(result["raw_tags"], 1):
                        st.write(f"{i}. {tag}")

                with cols[1]:
                    st.markdown("**Lexical**")
                    for i, tag in enumerate(result["lexical_tags"], 1):
                        st.write(f"{i}. {tag}")

                with cols[2]:
                    st.markdown("**Trend-aware**")
                    for i, tag in enumerate(result["trend_tags"], 1):
                        st.write(f"{i}. {tag}")

                with cols[3]:
                    st.markdown(f"**{result['final_name']}**")
                    for i, tag in enumerate(result["final_tags"], 1):
                        st.write(f"{i}. {tag}")

            with st.expander("Show scoring breakdown"):
                breakdown_df = pd.DataFrame(result["final_rows"][:top_k])

                if result["final_mode"] == "hybrid":
                    display_cols = [
                        "tag",
                        "final_score",
                        "base_score",
                        "sem_score",
                        "trend_score",
                        "cat_score",
                        "lex_score",
                        "penalty"
                    ]
                else:
                    display_cols = [
                        "tag",
                        "final_score",
                        "base_score",
                        "trend_score",
                        "cat_score",
                        "lex_score",
                        "penalty"
                    ]

                st.dataframe(
                    breakdown_df[display_cols],
                    use_container_width=True,
                    hide_index=True
                )


# =========================================================
# TAB 2: QUALITATIVE DEMO
# =========================================================
with tab2:
    st.subheader("Qualitative Comparison")

    selected_sample = st.selectbox(
        "Choose demo sample",
        options=list(range(len(SAMPLE_CAPTIONS))),
        format_func=lambda i: f"{SAMPLE_CAPTIONS[i][1]} — {SAMPLE_CAPTIONS[i][0][:50]}..."
    )

    sample_caption, sample_category = SAMPLE_CAPTIONS[selected_sample]

    st.markdown(f"**Caption:** {sample_caption}")
    st.markdown(f"**Category:** {sample_category}")

    compare_data = {
        "Rank": [1, 2, 3, 4, 5],
        "Final Model 1": run_model_family(
            system,
            "Final Model 1 - SVM + TF-IDF Semantic",
            sample_caption,
            sample_category,
            top_k=5,
            candidate_pool=20
        )["final_tags"],
        "Final Model 2": run_model_family(
            system,
            "Final Model 2 - SVM + BERT Semantic",
            sample_caption,
            sample_category,
            top_k=5,
            candidate_pool=20
        )["final_tags"],
    }

    if "SBERT + LR" in system["models"]:
        compare_data["SBERT + LR"] = run_model_family(
            system,
            "SBERT + LR",
            sample_caption,
            sample_category,
            top_k=5,
            candidate_pool=20
        )["final_tags"]

    compare_df = pd.DataFrame(compare_data)

    st.dataframe(
        compare_df,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# TAB 3: EVALUATION DEMO
# =========================================================
with tab3:
    st.subheader("Evaluation Demo")

    st.caption(
        "This table compares the final outputs of the deployed model families."
    )

    if st.button("Run Evaluation Table"):
        with st.spinner("Running evaluation..."):
            results_df = run_demo_evaluation(system)

        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("**Interpretation**")
        st.write(
            "The evaluation compares Final Model 1, Final Model 2, and SBERT + LR "
            "using ranking-based metrics such as LRAP, Precision@K, MAP@5, and NDCG@5."
        )
