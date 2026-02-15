import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="Adult Income Classifier", layout="wide")

st.title("🧠 Adult Income Classification System")
st.write("This application compares multiple machine learning classifiers for income prediction.")

# -------------------------------
# Load performance results
# -------------------------------
results = pd.read_csv("model_results.csv")

# -------------------------------
# Show comparison of all models
# -------------------------------
st.subheader("📊 Comparison of All Models")
st.dataframe(results, use_container_width=True)

st.markdown("---")

# -------------------------------
# Sidebar model selection
# -------------------------------
st.sidebar.header("Model Selection")
model_name = st.sidebar.selectbox(
    "Choose a classifier",
    results["Model"].values
)

# -------------------------------
# Show selected model metrics
# -------------------------------
st.subheader("📌 Selected Model Performance")
selected_row = results[results["Model"] == model_name]
st.dataframe(selected_row, use_container_width=True)

st.markdown("---")

# -------------------------------
# Upload dataset
# -------------------------------
st.subheader("📂 Upload Test Dataset")
uploaded_file = st.file_uploader("Upload CSV file containing test data", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("Preview of Uploaded Data:")
    st.dataframe(data.head())

    # Encode categorical columns
    for col in data.select_dtypes(include='object').columns:
        data[col] = data[col].astype('category').cat.codes

    X = data.drop("income", axis=1)
    y = data["income"]

    model = joblib.load(f"model/{model_name.replace(' ', '_')}.pkl")
    predictions = model.predict(X)

    # -------------------------------
    # Classification Report
    # -------------------------------
    st.subheader("📄 Classification Report")
    report = classification_report(y, predictions, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)

    # -------------------------------
    # Confusion Matrix
    # -------------------------------
    st.subheader("🔍 Confusion Matrix")
    cm = confusion_matrix(y, predictions)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {model_name}")

    st.pyplot(fig)
