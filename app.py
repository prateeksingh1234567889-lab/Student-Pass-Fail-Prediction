
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Student Pass/Fail Prediction",
    page_icon="🎓"
)

st.title("🎓 Student Pass/Fail Prediction")
st.write(
    "Predict student results using Machine Learning."
)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("students.csv")

try:
    df = load_data()
except FileNotFoundError:
    st.error("students.csv not found. Check your project folder.")
    st.stop()

# Input features and target
features = [
    "attendance",
    "internal_marks",
    "test_score",
    "assignment_score"
]

if not set(features + ["pass"]).issubset(df.columns):
    st.error("Dataset columns are missing or incorrect.")
    st.stop()

if df.empty or df[features + ["pass"]].isnull().any().any():
    st.error("Dataset is empty or contains missing values.")
    st.stop()

X = df[features]
y = df["pass"].astype(int)

if not set(y.unique()).issubset({0, 1}) or y.nunique() != 2:
    st.error("The pass column must contain both 0 and 1.")
    st.stop()

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Create three ML models
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42)
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=4,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

# Train and evaluate models
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results[name] = round(
        accuracy_score(y_test, predictions) * 100, 2
    )

st.subheader("📊 Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)
st.caption("This project uses a small illustrative dataset.")

st.subheader("📈 Model Accuracy Comparison")

comparison = pd.DataFrame({
    "Algorithm": list(results.keys()),
    "Test Accuracy (%)": list(results.values())
})

st.dataframe(comparison, hide_index=True)

selected_name = st.selectbox(
    "Choose Machine Learning Algorithm",
    list(models.keys())
)

st.subheader("📝 Enter Student Details")

attendance = st.slider(
    "Attendance (%)", 0, 100, 80
)

internal_marks = st.slider(
    "Internal Marks (out of 20)", 0, 20, 14
)

test_score = st.slider(
    "Test Score (out of 100)", 0, 100, 65
)

assignment_score = st.slider(
    "Assignment Score (out of 100)", 0, 100, 70
)

if st.button("Predict Result"):
    student = pd.DataFrame([{
        "attendance": attendance,
        "internal_marks": internal_marks,
        "test_score": test_score,
        "assignment_score": assignment_score
    }])

    model = models[selected_name]
    prediction = int(model.predict(student)[0])

    probabilities = model.predict_proba(student)[0]
    pass_index = list(model.classes_).index(1)
    pass_probability = probabilities[pass_index]

    if prediction == 1:
        st.success("Predicted Result: PASS ✅")
    else:
        st.error("Predicted Result: FAIL / AT RISK ⚠️")

    st.write(
        f"Estimated pass probability: {pass_probability:.1%}"
    )

    st.caption(
        "Educational prediction only; this is not an official "
        "academic result. The sample dataset is too small for "
        "reliable real-world predictions."
    )

st.subheader("About the Project")
st.write(
    "Algorithms used: Logistic Regression, Decision Tree "
    "and Random Forest."
)
st.write(
    "Libraries: Python, Pandas, NumPy, Scikit-learn and Streamlit."
)
