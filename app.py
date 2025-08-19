
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="ML Model with Streamlit", page_icon="🤖", layout="wide")

DATA_PATH = Path("data/dataset.csv")
MODEL_PATH = Path("model.pkl")
METRICS_PATH = Path("metrics.json")
COMPARE_PATH = Path("model_compare.json")
TEST_PATH = Path("data/test.csv")
MODEL_INFO_PATH = Path("model_info.json")

@st.cache_data
def load_data():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        return df
    # Fallback: build Iris dataset if data file is missing
    try:
        from sklearn.datasets import load_iris
        iris = load_iris(as_frame=True)
        df = iris.frame
        df.rename(columns={"target": "target"}, inplace=True)
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        return df
    except Exception as e:
        st.error(f"Could not load dataset: {e}")
        return None

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None

def page_home():
    st.title("🤖 Machine Learning Demo App")
    st.write("""
    This Streamlit app demonstrates a complete ML workflow:
    - Data exploration & visualizations
    - Model predictions with confidence
    - Model performance and comparison
    """)
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy (test)", f"{metrics.get('best_test_accuracy', 0):.3f}")
        col2.metric("CV Mean (best)", f"{metrics.get('best_cv_mean', 0):.3f}")
        col3.metric("Model", metrics.get("best_model_name", "N/A"))
    else:
        st.info("No metrics found yet. Train the model locally by running `python train.py`.")

def page_explore(df: pd.DataFrame):
    st.header("🔎 Data Exploration")
    if df is None:
        st.warning("Dataset not found. Please add data/dataset.csv or run train.py.")
        return
    st.subheader("Overview")
    c1, c2, c3 = st.columns(3)
    c1.write(f"**Rows:** {df.shape[0]}")
    c2.write(f"**Columns:** {df.shape[1]}")
    c3.write("**Dtypes:**")
    c3.write(df.dtypes.astype(str))
    st.subheader("Sample")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Interactive Filter")
    with st.expander("Filter rows"):
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        filters = {}
        for col in numeric_cols:
            mn, mx = float(df[col].min()), float(df[col].max())
            val = st.slider(col, min_value=mn, max_value=mx, value=(mn, mx))
            filters[col] = val
        mask = np.ones(len(df), dtype=bool)
        for col, (mn, mx) in filters.items():
            mask &= (df[col] >= mn) & (df[col] <= mx)
        st.write(f"Filtered rows: {mask.sum()}")
        st.dataframe(df.loc[mask], use_container_width=True)

def page_visualize(df: pd.DataFrame):
    st.header("📊 Visualizations")
    if df is None:
        st.warning("Dataset not found. Please add data/dataset.csv or run train.py.")
        return
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        st.warning("No numeric columns to visualize.")
        return

    st.subheader("Histogram")
    col = st.selectbox("Column", numeric_cols, key="hist_col")
    bins = st.slider("Bins", 5, 60, 20, key="bins")
    fig = px.histogram(df, x=col, nbins=bins, opacity=0.8)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Scatter Plot")
    c1, c2 = st.columns(2)
    x_col = c1.selectbox("X", numeric_cols, key="xcol")
    y_col = c2.selectbox("Y", numeric_cols, key="ycol")
    color_col = st.selectbox("Color (optional)", [None] + df.columns.tolist(), key="colorcol")
    fig2 = px.scatter(df, x=x_col, y=y_col, color=color_col if color_col else None)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Box Plot")
    box_col = st.selectbox("Box Column", numeric_cols, key="boxcol")
    fig3 = px.box(df, y=box_col, points="all")
    st.plotly_chart(fig3, use_container_width=True)

def page_predict(df: pd.DataFrame, model):
    st.header("🧮 Predict")
    if model is None:
        st.info("Model not found. Run `python train.py` to create model.pkl.")
        return
    if df is None:
        st.warning("Dataset missing; some defaults may not appear.")

    # Build inputs for Iris features by default; otherwise, infer numeric feature columns
    feature_names = None
    if MODEL_INFO_PATH.exists():
        with open(MODEL_INFO_PATH) as f:
            info = json.load(f)
            feature_names = info.get("feature_names")
    if feature_names is None:
        # Fallback: all numeric columns except target
        if df is not None:
            feature_names = [c for c in df.select_dtypes(include=["number"]).columns if c != "target"]
        else:
            feature_names = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]

    st.write("Enter feature values:")
    inputs = []
    cols = st.columns(2)
    for i, feat in enumerate(feature_names):
        # Use data-driven sensible defaults if df is available
        if df is not None and feat in df.columns and pd.api.types.is_numeric_dtype(df[feat]):
            mn, mx = float(df[feat].min()), float(df[feat].max())
            default = float(df[feat].median())
            val = cols[i % 2].number_input(feat, value=default, min_value=mn, max_value=mx, format="%.3f")
        else:
            val = cols[i % 2].number_input(feat, value=0.0, format="%.3f")
        inputs.append(val)

    if st.button("Predict"):
        try:
            with st.spinner("Predicting..."):
                arr = np.array(inputs).reshape(1, -1)
                pred = model.predict(arr)[0]
                proba = getattr(model, "predict_proba", None)
                if proba is not None:
                    probs = proba(arr)[0]
                    st.success(f"Prediction: {pred}")
                    st.write("Confidence/Probabilities:")
                    st.write({str(i): float(p) for i, p in enumerate(probs)})
                else:
                    st.success(f"Prediction: {pred}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

def page_performance():
    st.header("📈 Model Performance")
    if not METRICS_PATH.exists():
        st.info("No metrics yet. Run `python train.py`.")
        return
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    st.subheader("Summary")
    st.json(metrics)
    if COMPARE_PATH.exists():
        with open(COMPARE_PATH) as f:
            comp = json.load(f)
        st.subheader("Model Comparison")
        comp_df = pd.DataFrame(comp)
        st.dataframe(comp_df, use_container_width=True)
        if "model" in comp_df and "cv_mean" in comp_df:
            fig = px.bar(comp_df, x="model", y="cv_mean")
            st.plotly_chart(fig, use_container_width=True)
    # Confusion matrix if test set available
    if TEST_PATH.exists() and Path("model.pkl").exists():
        try:
            test_df = pd.read_csv(TEST_PATH)
            X = test_df.drop(columns=["target"])
            y = test_df["target"]
            with open("model.pkl", "rb") as f:
                model = pickle.load(f)
            from sklearn.metrics import confusion_matrix
            import plotly.figure_factory as ff
            y_pred = model.predict(X)
            cm = confusion_matrix(y, y_pred)
            z = cm
            x = [str(i) for i in sorted(np.unique(y))]
            ytick = [str(i) for i in sorted(np.unique(y))]
            fig = ff.create_annotated_heatmap(z, x=x, y=ytick, showscale=True)
            st.subheader("Confusion Matrix (Test Set)")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render confusion matrix: {e}")

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Explore Data", "Visualizations", "Predict", "Model Performance", "About"])
    df = load_data()
    model = load_model()
    if page == "Home":
        page_home()
    elif page == "Explore Data":
        page_explore(df)
    elif page == "Visualizations":
        page_visualize(df)
    elif page == "Predict":
        page_predict(df, model)
    elif page == "Model Performance":
        page_performance()
    else:
        st.header("ℹ️ About")
        st.write("This app is a template. Replace the dataset and retrain the model for your chosen problem.")
        st.write("Built with Streamlit, scikit-learn, and Plotly.")

if __name__ == "__main__":
    main()
