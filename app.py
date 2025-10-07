# import streamlit as st
# import pandas as pd
# import json
# import heapq
# from datetime import datetime, timezone
# from pathlib import Path
# import matplotlib.pyplot as plt
# import joblib
# import pickle

# # --------------------
# # Configuration
# # --------------------
# st.set_page_config(page_title="Sentiment-Based Feed Organizer", layout="wide")
# st.title("Sentiment-Based Feed Organizer")
# st.markdown(
#     "Load cleaned data, optionally re-run a saved model, merge with an existing feed JSON, "
#     "and visualize / explore the prioritized sentiment feed."
# )
# st.markdown("---")

# # File paths (as you specified)
# DATA_PATH = Path("data/labeled/fully_cleaned.csv")
# JSON_PATH = Path("data/processed/final_feed_20251005_231939.json")
# MODEL_PATH = Path("data/processed/tfidf_vectors.npz")
# # common TF-IDF vectorizer path used by training code (try to find it)
# VECTORIZER_PATHS = [
#     Path("data/processed/tfidf_vectorizer_lr.pkl"),
#     Path("data/processed/tfidf_model.pkl"),
#     Path("data/processed/tfidf_model_vectorizer.pkl")
# ]

# # --------------------
# # Session state helpers
# # --------------------
# if "cleaned_df" not in st.session_state:
#     st.session_state["cleaned_df"] = None
# if "json_feed_df" not in st.session_state:
#     st.session_state["json_feed_df"] = None
# if "merged_df" not in st.session_state:
#     st.session_state["merged_df"] = None
# if "final_feed_df" not in st.session_state:
#     st.session_state["final_feed_df"] = None
# if "model" not in st.session_state:
#     st.session_state["model"] = None
# if "vectorizer" not in st.session_state:
#     st.session_state["vectorizer"] = None


# # --------------------
# # Utility functions
# # --------------------
# def load_cleaned_csv(path: Path) -> pd.DataFrame:
#     """Load the cleaned CSV and normalize columns we need."""
#     df = pd.read_csv(path, low_memory=False)
#     # Ensure id is string
#     if "id" in df.columns:
#         df["id"] = df["id"].astype(str)
#     # created_utc -> created_at (datetime)
#     if "created_utc" in df.columns:
#         # some pipelines store created_utc as epoch seconds
#         try:
#             df["created_at"] = pd.to_datetime(pd.to_numeric(df["created_utc"], errors="coerce"), unit="s", origin="unix")
#         except Exception:
#             df["created_at"] = pd.to_datetime(df["created_utc"], errors="coerce")
#     elif "created_at" in df.columns:
#         df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
#     else:
#         df["created_at"] = pd.NaT

#     # canonical sentiment and confidence column names
#     # your CSV uses 'sentiment' and 'sentiment_confidence'
#     if "sentiment_confidence" in df.columns and "confidence" not in df.columns:
#         df["confidence"] = pd.to_numeric(df["sentiment_confidence"], errors="coerce").fillna(0.0)
#     elif "confidence" not in df.columns:
#         df["confidence"] = 0.0

#     # cleaned_text exists per your description
#     if "cleaned_text" not in df.columns:
#         st.warning("CSV missing 'cleaned_text' column; model prediction may fail.")
#         df["cleaned_text"] = df.get("raw_text", "").astype(str)

#     return df


# def load_json_feed(path: Path) -> pd.DataFrame:
#     """
#     Load the JSON final_feed file where keys are sentiment categories.
#     Returns a dataframe with columns: id, json_text, json_confidence, json_timestamp, priority_score, sentiment
#     """
#     with open(path, "r", encoding="utf-8") as fh:
#         obj = json.load(fh)

#     rows = []
#     for sentiment_cat, items in obj.items():
#         for it in items:
#             row = {
#                 "id": str(it.get("id", "")),
#                 "json_text": it.get("text", ""),
#                 "json_confidence": float(it.get("confidence", 0.0)),
#                 "json_timestamp": it.get("timestamp", None),
#                 "priority_score": float(it.get("priority_score", 0.0)),
#                 "json_sentiment": sentiment_cat.lower()
#             }
#             rows.append(row)
#     if not rows:
#         return pd.DataFrame(rows)

#     df = pd.DataFrame(rows)
#     # parse timestamp strings to datetime
#     df["json_timestamp"] = pd.to_datetime(df["json_timestamp"], errors="coerce")
#     # keep canonical names for merge convenience
#     df["id"] = df["id"].astype(str)
#     return df


# def merge_cleaned_and_json(cleaned: pd.DataFrame, json_df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Merge on id. Preference for JSON-derived sentiment/confidence if present.
#     Output columns:
#       - id, subreddit, title, url, cleaned_text, created_at, sentiment, confidence, priority_score, json_text
#     """
#     if cleaned is None and json_df is None:
#         return None
#     if cleaned is None:
#         # Use json-only data
#         df = json_df.copy()
#         df["sentiment"] = df["json_sentiment"]
#         df["confidence"] = df["json_confidence"]
#         df["created_at"] = df["json_timestamp"]
#         return df

#     df = cleaned.copy()
#     if json_df is not None and not json_df.empty:
#         merged = pd.merge(df, json_df, on="id", how="left", suffixes=("", "_json"))
#     else:
#         merged = df

#     # If JSON sentiment exists use it, else fall back to CSV 'sentiment'
#     if "json_sentiment" in merged.columns:
#         merged["sentiment"] = merged["json_sentiment"].fillna(merged.get("sentiment", pd.Series(dtype="object")))
#     else:
#         merged["sentiment"] = merged.get("sentiment", pd.Series(dtype="object"))

#     # Confidence preference: json_confidence -> sentiment_confidence -> confidence -> default 0.0
#     merged["confidence"] = (
#         merged.get("json_confidence")
#         .fillna(merged.get("sentiment_confidence"))
#         .fillna(merged.get("confidence"))
#         .fillna(0.0)
#     )

#     # If missing created_at but json timestamp present, use that
#     if "created_at" not in merged.columns or merged["created_at"].isna().all():
#         if "json_timestamp" in merged.columns:
#             merged["created_at"] = merged["json_timestamp"]
#     # Ensure created_at is datetime
#     merged["created_at"] = pd.to_datetime(merged["created_at"], errors="coerce")

#     # fallback text column: prefer json_text if present else cleaned_text (cleaned_text should exist)
#     merged["display_text"] = merged.get("json_text").fillna(merged.get("cleaned_text", ""))
#     # fill missing sentiment values with 'neutral'
#     merged["sentiment"] = merged["sentiment"].fillna("neutral").str.lower()
#     # ensure confidence numeric
#     merged["confidence"] = pd.to_numeric(merged["confidence"], errors="coerce").fillna(0.0)

#     return merged


# def load_model_and_vectorizer(model_path: Path):
#     """
#     Try joblib.load first; fall back to pickle.load. Also attempt to find a vectorizer file.
#     Returns (model, vectorizer_or_none)
#     """
#     model = None
#     vectorizer = None
#     try:
#         model = joblib.load(model_path)
#     except Exception:
#         try:
#             with open(model_path, "rb") as fh:
#                 model = pickle.load(fh)
#         except Exception as e:
#             st.error(f"Unable to load model at {model_path}: {e}")
#             return None, None

#     # Try to load vectorizer from known paths
#     for vp in VECTORIZER_PATHS:
#         if vp.exists():
#             try:
#                 vectorizer = joblib.load(vp)
#                 break
#             except Exception:
#                 # try pickle
#                 try:
#                     with open(vp, "rb") as fh:
#                         vectorizer = pickle.load(fh)
#                         break
#                 except Exception:
#                     continue

#     return model, vectorizer


# def model_predict_apply(model, vectorizer, texts):
#     """Run predictions on a sequence of texts. Return (labels, confidences)."""
#     # texts: pandas Series
#     if texts is None or len(texts) == 0:
#         return [], []

#     # If model is a sklearn pipeline that includes vectorizer, we can call model.predict directly on raw texts.
#     try:
#         # Preferred path: model accepts raw texts and supports predict_proba
#         preds = model.predict(texts.tolist())
#         confidences = None
#         if hasattr(model, "predict_proba"):
#             probs = model.predict_proba(texts.tolist())
#             confidences = [max(p) for p in probs]
#         else:
#             # fallback: try decision_function to estimate confidence
#             if hasattr(model, "decision_function"):
#                 df_vals = model.decision_function(texts.tolist())
#                 # If binary, convert to probability-like score
#                 import numpy as np
#                 if len(df_vals.shape) == 1:
#                     scores = 1 / (1 + np.exp(-df_vals))  # sigmoid
#                     confidences = [float(abs(s)) for s in scores]
#                 else:
#                     confidences = [float(max(1 / (1 + np.exp(-v)).max(), 0.5)) for v in df_vals]
#             else:
#                 # no confidence available, fallback to 0.5
#                 confidences = [0.5] * len(preds)
#         return list(map(str, preds)), list(map(float, confidences))
#     except Exception:
#         # Try transform via vectorizer -> model.predict
#         if vectorizer is not None:
#             try:
#                 X = vectorizer.transform(texts.fillna("").astype(str))
#                 preds = model.predict(X)
#                 if hasattr(model, "predict_proba"):
#                     probs = model.predict_proba(X)
#                     confidences = [max(p) for p in probs]
#                 else:
#                     confidences = [0.5] * len(preds)
#                 return list(map(str, preds)), list(map(float, confidences))
#             except Exception as e:
#                 st.error(f"Model prediction failed after using vectorizer: {e}")
#                 return [], []
#         else:
#             st.error("Model does not accept raw text and no vectorizer was found.")
#             return [], []


# def compute_priority_score_from_row(row, now=None):
#     """Compute priority = confidence * recency_factor (recency in hours)."""
#     if now is None:
#         now = datetime.now(timezone.utc)
#     created = row.get("created_at")
#     if pd.isna(created):
#         # fallback very low recency effect
#         recency_factor = 0.1
#     else:
#         # ensure timezone aware
#         try:
#             created_ts = pd.to_datetime(created)
#             if created_ts.tzinfo is None:
#                 created_ts = created_ts.tz_localize(None)
#             delta_hours = (now - created_ts).total_seconds() / 3600.0
#             # decay: 1/(1 + hours) bounded
#             recency_factor = max(1.0 / (1.0 + delta_hours), 0.01)
#         except Exception:
#             recency_factor = 0.1

#     confidence = float(row.get("confidence", 0.0)) if row.get("confidence", None) is not None else 0.0
#     return confidence * recency_factor


# def build_final_feed(merged_df: pd.DataFrame, json_feed_df: pd.DataFrame = None) -> pd.DataFrame:
#     """
#     If json_feed_df is present, use its priority_score and order (preferred).
#     Otherwise compute priority_score and sort.
#     Returns DataFrame with columns: id, title, subreddit, display_text, sentiment, confidence, created_at, priority_score, url
#     """
#     if json_feed_df is not None and not json_feed_df.empty:
#         # json_feed_df has id, json_text, json_confidence, json_timestamp, priority_score, json_sentiment
#         jf = json_feed_df.copy()
#         jf["priority_score"] = jf["priority_score"].astype(float)
#         jf["created_at"] = pd.to_datetime(jf["json_timestamp"], errors="coerce")
#         jf["display_text"] = jf["json_text"]
#         jf["sentiment"] = jf["json_sentiment"].fillna("neutral").str.lower()
#         jf["confidence"] = jf["json_confidence"].astype(float)
#         # merge with cleaned to bring title, subreddit, url if available
#         if merged_df is not None:
#             left = merged_df[["id", "title", "subreddit", "url"]].drop_duplicates(subset=["id"])
#             final = pd.merge(jf, left, on="id", how="left")
#         else:
#             final = jf
#         final = final.sort_values("priority_score", ascending=False).reset_index(drop=True)
#         return final

#     # else compute from merged_df
#     if merged_df is None or merged_df.empty:
#         return pd.DataFrame()
#     df = merged_df.copy()
#     df["priority_score"] = df.apply(lambda r: compute_priority_score_from_row(r), axis=1)
#     df = df.sort_values("priority_score", ascending=False).reset_index(drop=True)
#     return df


# # --------------------
# # Sidebar: Controls
# # --------------------
# st.sidebar.header("Data controls")
# if st.sidebar.button("Load cleaned CSV"):
#     if DATA_PATH.exists():
#         st.session_state["cleaned_df"] = load_cleaned_csv(DATA_PATH)
#         st.success(f"Loaded cleaned CSV: {len(st.session_state['cleaned_df'])} rows")
#     else:
#         st.error(f"Cleaned CSV not found at {DATA_PATH}")

# if st.sidebar.button("Load JSON feed"):
#     if JSON_PATH.exists():
#         st.session_state["json_feed_df"] = load_json_feed(JSON_PATH)
#         st.success(f"Loaded JSON feed: {len(st.session_state['json_feed_df'])} entries")
#         # if cleaned data already loaded, merge automatically
#         if st.session_state["cleaned_df"] is not None:
#             st.session_state["merged_df"] = merge_cleaned_and_json(st.session_state["cleaned_df"], st.session_state["json_feed_df"])
#             st.info("Merged JSON feed with cleaned CSV (merged_df available).")
#         else:
#             st.session_state["merged_df"] = merge_cleaned_and_json(None, st.session_state["json_feed_df"])
#     else:
#         st.error(f"JSON feed not found at {JSON_PATH}")

# if st.sidebar.button("Load model (optional)"):
#     if MODEL_PATH.exists():
#         model, vectorizer = load_model_and_vectorizer(MODEL_PATH)
#         if model is not None:
#             st.session_state["model"] = model
#             st.session_state["vectorizer"] = vectorizer
#             st.success("Model loaded.")
#             if vectorizer is not None:
#                 st.info("Vectorizer loaded alongside model.")
#         else:
#             st.error("Model load failed.")
#     else:
#         st.error(f"Model file not found at {MODEL_PATH}")

# if st.sidebar.button("Run model on cleaned data"):
#     if st.session_state["model"] is None:
#         st.error("Load model first (use 'Load model (optional)').")
#     elif st.session_state["cleaned_df"] is None:
#         st.error("Load cleaned CSV first.")
#     else:
#         model = st.session_state["model"]
#         vectorizer = st.session_state["vectorizer"]
#         cleaned = st.session_state["cleaned_df"]
#         texts = cleaned["cleaned_text"].fillna("").astype(str)
#         with st.spinner("Running model predictions (this may take time)..."):
#             labels, confidences = model_predict_apply(model, vectorizer, texts)
#         if labels:
#             cleaned["predicted_sentiment_from_model"] = labels
#             cleaned["predicted_confidence_from_model"] = confidences
#             # update merged_df (if existed) or set merged_df to cleaned
#             if st.session_state["merged_df"] is not None:
#                 st.session_state["merged_df"] = merge_cleaned_and_json(cleaned, st.session_state.get("json_feed_df"))
#             else:
#                 st.session_state["merged_df"] = merge_cleaned_and_json(cleaned, None)
#             st.success("Model predictions added and merged_df updated.")
#         else:
#             st.error("Model prediction failed or returned no labels.")

# if st.sidebar.button("Generate final feed"):
#     st.session_state["final_feed_df"] = build_final_feed(st.session_state.get("merged_df"), st.session_state.get("json_feed_df"))
#     if st.session_state["final_feed_df"] is None or st.session_state["final_feed_df"].empty:
#         st.warning("Final feed is empty. Ensure you loaded CSV or JSON feed first.")
#     else:
#         st.success(f"Final feed ready — {len(st.session_state['final_feed_df'])} posts")

# if st.sidebar.button("Clear loaded data"):
#     st.session_state["cleaned_df"] = None
#     st.session_state["json_feed_df"] = None
#     st.session_state["merged_df"] = None
#     st.session_state["final_feed_df"] = None
#     st.session_state["model"] = None
#     st.session_state["vectorizer"] = None
#     st.success("Session cleared.")


# # --------------------
# # Main view -- show loaded/merged data
# # --------------------
# st.header("Data preview and controls")

# col1, col2 = st.columns([1, 1])
# with col1:
#     st.subheader("Cleaned CSV (data/labeled/fully_cleaned.csv)")
#     if st.session_state["cleaned_df"] is not None:
#         st.write(f"Rows: {len(st.session_state['cleaned_df'])}")
#         st.dataframe(st.session_state["cleaned_df"].head(5))
#     else:
#         st.info("Cleaned CSV not loaded. Click 'Load cleaned CSV' in the sidebar.")

# with col2:
#     st.subheader("JSON Feed (data/processed/...json)")
#     if st.session_state["json_feed_df"] is not None:
#         st.write(f"Entries: {len(st.session_state['json_feed_df'])}")
#         st.dataframe(st.session_state["json_feed_df"].head(5))
#     else:
#         st.info("JSON feed not loaded. Click 'Load JSON feed' in the sidebar.")


# st.markdown("---")
# st.subheader("Merged dataset (merged_df)")
# if st.session_state["merged_df"] is not None:
#     st.write(f"Rows: {len(st.session_state['merged_df'])}")
#     st.dataframe(
#         st.session_state["merged_df"][
#             ["id", "title", "subreddit", "sentiment", "confidence", "created_at"]
#         ].head(10)
#     )
# else:
#     st.info("merged_df not available. Load CSV and/or JSON feed and they will be merged automatically.")


# # --------------------
# # Feed viewer
# # --------------------
# st.markdown("---")
# st.header("Final feed viewer")

# if st.session_state["final_feed_df"] is None:
#     st.info("Final feed not generated yet. Use 'Generate final feed' in the sidebar.")
# else:
#     ff = st.session_state["final_feed_df"]
#     st.write(f"Displayed posts: {len(ff)}")
#     # filter by sentiment
#     sentiments_available = sorted(ff["sentiment"].dropna().unique().tolist())
#     sentiment_filter = st.selectbox("Filter by sentiment", ["All"] + sentiments_available)
#     df_display = ff if sentiment_filter == "All" else ff[ff["sentiment"] == sentiment_filter]

#     # show top-K controls
#     top_k = st.number_input("Show top K posts", min_value=1, max_value=min(500, max(10, len(df_display))), value=50)
#     df_display = df_display.head(top_k)

#     for i, row in df_display.iterrows():
#         st.markdown(f"**{i+1}. {row.get('title', 'Untitled')}**")
#         st.markdown(f"- Subreddit: {row.get('subreddit', 'N/A')}")
#         st.markdown(f"- Sentiment: **{row.get('sentiment', 'unknown')}** (confidence: {float(row.get('confidence', 0)):.3f})")
#         st.markdown(f"- Priority score: {float(row.get('priority_score', row.get('priority_score', 0))):.5f}")
#         if row.get("display_text"):
#             text_preview = str(row.get("display_text"))
#             if len(text_preview) > 500:
#                 text_preview = text_preview[:500] + "..."
#             st.markdown(f"> {text_preview}")
#         if row.get("url"):
#             st.markdown(f"[Open original post]({row.get('url')})")
#         st.markdown("---")


# # --------------------
# # Analytics
# # --------------------
# st.markdown("---")
# st.header("Analytics")

# merged_for_analytics = st.session_state["merged_df"]
# if merged_for_analytics is None or merged_for_analytics.empty:
#     st.info("No data to show analytics. Load and merge data first.")
# else:
#     col1, col2, col3 = st.columns([1, 1, 1])
#     with col1:
#         total = len(merged_for_analytics)
#         st.metric("Total posts", total)
#     with col2:
#         pos = int((merged_for_analytics["sentiment"] == "positive").sum())
#         st.metric("Positive", pos)
#     with col3:
#         neg = int((merged_for_analytics["sentiment"] == "negative").sum())
#         st.metric("Negative", neg)

#     # sentiment distribution pie
#     counts = merged_for_analytics["sentiment"].value_counts()
#     fig1, ax1 = plt.subplots(figsize=(4, 3))
#     ax1.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
#     ax1.set_title("Sentiment distribution")
#     st.pyplot(fig1)

#     # top subreddits by count
#     top_subs = merged_for_analytics["subreddit"].value_counts().head(10)
#     fig2, ax2 = plt.subplots(figsize=(6, 3))
#     top_subs.plot(kind="barh", ax=ax2)
#     ax2.invert_yaxis()
#     ax2.set_xlabel("Posts")
#     ax2.set_title("Top subreddits (by post count)")
#     st.pyplot(fig2)

import streamlit as st
import pandas as pd
import json
from datetime import datetime, timezone
from pathlib import Path
import matplotlib.pyplot as plt
import joblib
import pickle

# --------------------
# Configuration
# --------------------
st.set_page_config(page_title="Sentiment-Based Feed Organizer", layout="wide")
st.title("🎯 Sentiment-Based Feed Organizer")
st.markdown(
    "Load cleaned data, optionally re-run a saved model, merge with an existing feed JSON, "
    "and visualize / explore the prioritized sentiment feed."
)
st.markdown("---")

# File paths
DATA_PATH = Path("data/labeled/fully_cleaned.csv")
JSON_PATH = Path("data/processed/final_feed_20251005_231939.json")
MODEL_PATH = Path("data/processed/tfidf_vectors.npz")
VECTORIZER_PATHS = [
    Path("data/processed/tfidf_vectorizer_lr.pkl"),
    Path("data/processed/tfidf_model.pkl"),
    Path("data/processed/tfidf_model_vectorizer.pkl")
]

# --------------------
# Session state initialization
# --------------------
if "cleaned_df" not in st.session_state:
    st.session_state["cleaned_df"] = None
if "json_feed_df" not in st.session_state:
    st.session_state["json_feed_df"] = None
if "merged_df" not in st.session_state:
    st.session_state["merged_df"] = None
if "final_feed_df" not in st.session_state:
    st.session_state["final_feed_df"] = None
if "model" not in st.session_state:
    st.session_state["model"] = None
if "vectorizer" not in st.session_state:
    st.session_state["vectorizer"] = None


# --------------------
# Utility functions
# --------------------
def load_cleaned_csv(path: Path) -> pd.DataFrame:
    """Load the cleaned CSV and normalize columns."""
    df = pd.read_csv(path, low_memory=False)
    
    # Ensure id is string
    if "id" in df.columns:
        df["id"] = df["id"].astype(str)
    
    # Handle created_utc -> created_at conversion
    if "created_utc" in df.columns:
        try:
            df["created_at"] = pd.to_datetime(pd.to_numeric(df["created_utc"], errors="coerce"), unit="s", origin="unix")
        except Exception:
            df["created_at"] = pd.to_datetime(df["created_utc"], errors="coerce")
    elif "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    else:
        df["created_at"] = pd.NaT

    # Handle confidence column
    if "sentiment_confidence" in df.columns and "confidence" not in df.columns:
        df["confidence"] = pd.to_numeric(df["sentiment_confidence"], errors="coerce").fillna(0.0)
    elif "confidence" not in df.columns:
        df["confidence"] = 0.0

    # Ensure cleaned_text exists
    if "cleaned_text" not in df.columns:
        st.warning("CSV missing 'cleaned_text' column; using raw_text if available.")
        df["cleaned_text"] = df.get("raw_text", "").astype(str)

    return df


def load_json_feed(path: Path) -> pd.DataFrame:
    """Load the JSON final_feed file where keys are sentiment categories."""
    with open(path, "r", encoding="utf-8") as fh:
        obj = json.load(fh)

    rows = []
    for sentiment_cat, items in obj.items():
        for it in items:
            row = {
                "id": str(it.get("id", "")),
                "json_text": it.get("text", ""),
                "json_confidence": float(it.get("confidence", 0.0)),
                "json_timestamp": it.get("timestamp", None),
                "priority_score": float(it.get("priority_score", 0.0)),
                "json_sentiment": sentiment_cat.lower()
            }
            rows.append(row)
    
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["json_timestamp"] = pd.to_datetime(df["json_timestamp"], errors="coerce")
    df["id"] = df["id"].astype(str)
    return df


def merge_cleaned_and_json(cleaned: pd.DataFrame, json_df: pd.DataFrame) -> pd.DataFrame:
    """Merge cleaned CSV and JSON feed data."""
    if cleaned is None and json_df is None:
        return None
    if cleaned is None:
        df = json_df.copy()
        df["sentiment"] = df["json_sentiment"]
        df["confidence"] = df["json_confidence"]
        df["created_at"] = df["json_timestamp"]
        return df

    df = cleaned.copy()
    if json_df is not None and not json_df.empty:
        merged = pd.merge(df, json_df, on="id", how="left", suffixes=("", "_json"))
    else:
        merged = df

    # Merge sentiment data
    if "json_sentiment" in merged.columns:
        merged["sentiment"] = merged["json_sentiment"].fillna(merged.get("sentiment", pd.Series(dtype="object")))
    else:
        merged["sentiment"] = merged.get("sentiment", pd.Series(dtype="object"))

    # Merge confidence data
    merged["confidence"] = (
        merged.get("json_confidence")
        .fillna(merged.get("sentiment_confidence"))
        .fillna(merged.get("confidence"))
        .fillna(0.0)
    )

    # Handle created_at
    if "created_at" not in merged.columns or merged["created_at"].isna().all():
        if "json_timestamp" in merged.columns:
            merged["created_at"] = merged["json_timestamp"]
    merged["created_at"] = pd.to_datetime(merged["created_at"], errors="coerce")

    # Display text preference
    merged["display_text"] = merged.get("json_text").fillna(merged.get("cleaned_text", ""))
    merged["sentiment"] = merged["sentiment"].fillna("neutral").str.lower()
    merged["confidence"] = pd.to_numeric(merged["confidence"], errors="coerce").fillna(0.0)

    return merged


def compute_priority_score_from_row(row, now=None):
    """Compute priority = confidence * recency_factor."""
    if now is None:
        now = datetime.now(timezone.utc)
    
    created = row.get("created_at")
    if pd.isna(created):
        recency_factor = 0.1
    else:
        try:
            created_ts = pd.to_datetime(created)
            if created_ts.tzinfo is None:
                created_ts = created_ts.replace(tzinfo=timezone.utc)
            delta_hours = (now - created_ts).total_seconds() / 3600.0
            recency_factor = max(1.0 / (1.0 + delta_hours), 0.01)
        except Exception:
            recency_factor = 0.1

    confidence = float(row.get("confidence", 0.0)) if row.get("confidence", None) is not None else 0.0
    return confidence * recency_factor


def build_final_feed(merged_df: pd.DataFrame, json_feed_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Build final feed with priority scores and sorting.
    Returns DataFrame with all necessary columns including URL for Reddit posts.
    """
    if json_feed_df is not None and not json_feed_df.empty:
        jf = json_feed_df.copy()
        jf["priority_score"] = jf["priority_score"].astype(float)
        jf["created_at"] = pd.to_datetime(jf["json_timestamp"], errors="coerce")
        jf["display_text"] = jf["json_text"]
        jf["sentiment"] = jf["json_sentiment"].fillna("neutral").str.lower()
        jf["confidence"] = jf["json_confidence"].astype(float)
        
        # Merge with cleaned to bring title, subreddit, url
        if merged_df is not None:
            left = merged_df[["id", "title", "subreddit", "url"]].drop_duplicates(subset=["id"])
            final = pd.merge(jf, left, on="id", how="left")
        else:
            final = jf
        
        final = final.sort_values("priority_score", ascending=False).reset_index(drop=True)
        return final

    # Compute from merged_df
    if merged_df is None or merged_df.empty:
        return pd.DataFrame()
    
    df = merged_df.copy()
    df["priority_score"] = df.apply(lambda r: compute_priority_score_from_row(r), axis=1)
    df = df.sort_values("priority_score", ascending=False).reset_index(drop=True)
    return df


# --------------------
# Sidebar Controls
# --------------------
st.sidebar.header("📂 Data Controls")

if st.sidebar.button("📥 Load cleaned CSV"):
    if DATA_PATH.exists():
        st.session_state["cleaned_df"] = load_cleaned_csv(DATA_PATH)
        st.sidebar.success(f"✅ Loaded {len(st.session_state['cleaned_df'])} rows")
    else:
        st.sidebar.error(f"❌ File not found: {DATA_PATH}")

if st.sidebar.button("📥 Load JSON feed"):
    if JSON_PATH.exists():
        st.session_state["json_feed_df"] = load_json_feed(JSON_PATH)
        st.sidebar.success(f"✅ Loaded {len(st.session_state['json_feed_df'])} entries")
        
        # Auto-merge if cleaned data exists
        if st.session_state["cleaned_df"] is not None:
            st.session_state["merged_df"] = merge_cleaned_and_json(
                st.session_state["cleaned_df"], 
                st.session_state["json_feed_df"]
            )
            st.sidebar.info("🔄 Auto-merged with cleaned CSV")
        else:
            st.session_state["merged_df"] = merge_cleaned_and_json(None, st.session_state["json_feed_df"])
    else:
        st.sidebar.error(f"❌ File not found: {JSON_PATH}")

if st.sidebar.button("🚀 Generate final feed"):
    st.session_state["final_feed_df"] = build_final_feed(
        st.session_state.get("merged_df"), 
        st.session_state.get("json_feed_df")
    )
    if st.session_state["final_feed_df"] is None or st.session_state["final_feed_df"].empty:
        st.sidebar.warning("⚠️ Final feed is empty. Load data first.")
    else:
        st.sidebar.success(f"✅ Generated feed with {len(st.session_state['final_feed_df'])} posts")

if st.sidebar.button("🗑️ Clear all data"):
    st.session_state["cleaned_df"] = None
    st.session_state["json_feed_df"] = None
    st.session_state["merged_df"] = None
    st.session_state["final_feed_df"] = None
    st.session_state["model"] = None
    st.session_state["vectorizer"] = None
    st.sidebar.success("🧹 Session cleared")


# --------------------
# Main View - Data Preview
# --------------------
st.header("📊 Data Preview")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Cleaned CSV Data")
    if st.session_state["cleaned_df"] is not None:
        st.info(f"📈 Total rows: {len(st.session_state['cleaned_df'])}")
        st.dataframe(st.session_state["cleaned_df"].head(5), use_container_width=True)
    else:
        st.warning("⚠️ No CSV data loaded")

with col2:
    st.subheader("JSON Feed Data")
    if st.session_state["json_feed_df"] is not None:
        st.info(f"📈 Total entries: {len(st.session_state['json_feed_df'])}")
        st.dataframe(st.session_state["json_feed_df"].head(5), use_container_width=True)
    else:
        st.warning("⚠️ No JSON data loaded")


# --------------------
# Final Feed Viewer (Main Feature)
# --------------------
st.markdown("---")
st.header("🎯 Final Feed Viewer")

if st.session_state["final_feed_df"] is None or st.session_state["final_feed_df"].empty:
    st.info("ℹ️ Click '🚀 Generate final feed' in the sidebar to create your feed")
else:
    ff = st.session_state["final_feed_df"]
    
    # Filters
    col1, col2 = st.columns([2, 1])
    with col1:
        sentiments_available = sorted(ff["sentiment"].dropna().unique().tolist())
        sentiment_filter = st.selectbox("🎨 Filter by sentiment:", ["All"] + sentiments_available)
    with col2:
        top_k = st.number_input("📊 Show top K posts:", min_value=1, max_value=500, value=50)
    
    # Apply filters
    df_display = ff if sentiment_filter == "All" else ff[ff["sentiment"] == sentiment_filter]
    df_display = df_display.head(top_k)
    
    st.info(f"Displaying **{len(df_display)}** posts")
    st.markdown("---")
    
    # Display posts with proper formatting (matching your image)
    for idx, row in df_display.iterrows():
        # Title with numbering
        st.markdown(f"### {idx+1}. {row.get('title', 'Untitled')}")
        
        # Metadata in columns
        meta_col1, meta_col2, meta_col3 = st.columns(3)
        with meta_col1:
            st.markdown(f"**Subreddit:** r/{row.get('subreddit', 'Unknown')}")
        with meta_col2:
            sentiment_emoji = {"positive": "😊", "negative": "😟", "neutral": "😐"}.get(row.get('sentiment', 'neutral'), "❓")
            st.markdown(f"**Sentiment:** {sentiment_emoji} {row.get('sentiment', 'unknown').capitalize()}")
        with meta_col3:
            st.markdown(f"**Confidence:** {float(row.get('confidence', 0)):.3f}")
        
        # Priority score
        st.markdown(f"**Priority Score:** {float(row.get('priority_score', 0)):.5f}")
        
        # Text preview
        if row.get("display_text"):
            text_preview = str(row.get("display_text", ""))
            if len(text_preview) > 400:
                text_preview = text_preview[:400] + "..."
            st.markdown(f"> {text_preview}")
        
        # URL to original Reddit post (THIS IS THE KEY FEATURE FROM YOUR IMAGE)
        if row.get("url") and pd.notna(row.get("url")):
            st.markdown(f"[🔗 Open original post]({row.get('url')})")
        else:
            st.markdown("*No URL available*")
        
        st.markdown("---")


# --------------------
# Analytics Section
# --------------------
st.markdown("---")
st.header("📈 Analytics Dashboard")

merged_for_analytics = st.session_state.get("merged_df")
if merged_for_analytics is None or merged_for_analytics.empty:
    st.info("ℹ️ Load and merge data to see analytics")
else:
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📝 Total Posts", len(merged_for_analytics))
    with col2:
        pos_count = int((merged_for_analytics["sentiment"] == "positive").sum())
        st.metric("😊 Positive", pos_count)
    with col3:
        neg_count = int((merged_for_analytics["sentiment"] == "negative").sum())
        st.metric("😟 Negative", neg_count)
    with col4:
        neu_count = int((merged_for_analytics["sentiment"] == "neutral").sum())
        st.metric("😐 Neutral", neu_count)
    
    # Charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("Sentiment Distribution")
        counts = merged_for_analytics["sentiment"].value_counts()
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        colors = {'positive': '#4CAF50', 'negative': '#F44336', 'neutral': '#9E9E9E'}
        pie_colors = [colors.get(label, '#757575') for label in counts.index]
        ax1.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90, colors=pie_colors)
        ax1.set_title("Sentiment Distribution")
        st.pyplot(fig1)
    
    with chart_col2:
        st.subheader("Top 10 Subreddits")
        top_subs = merged_for_analytics["subreddit"].value_counts().head(10)
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        top_subs.plot(kind="barh", ax=ax2, color='#2196F3')
        ax2.invert_yaxis()
        ax2.set_xlabel("Number of Posts")
        ax2.set_title("Top Subreddits by Post Count")
        st.pyplot(fig2)
        
        
        
# python -m streamlit run app.py