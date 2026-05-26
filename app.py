import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# page configiration
st.set_page_config(
    page_title="Body Type Predictor",
    page_icon="💪",
    layout="wide"
)

# css
st.markdown("""
<style>

.main {
    background-color: #0f172a;
}

h1 {
    color: white;
    text-align: center;
    font-size: 50px !important;
}

h3 {
    color: #cbd5e1;
}

.stButton>button {
    background: linear-gradient(90deg,#ff512f,#dd2476);
    color: white;
    border-radius: 12px;
    height: 55px;
    width: 100%;
    font-size: 22px;
    border: none;
    font-weight: bold;
}

.stButton>button:hover {
    transform: scale(1.02);
    transition: 0.3s;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    background-color: #1e293b;
    color: white;
    text-align: center;
    font-size: 30px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# reading dataset 
df = pd.read_csv("data.csv")

# input given to model
X = df[['Height', 'Weight', 'BMI', 'BodyFat']]

# output
y = df['Type']

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2, #20% testing , 80 % training 
    random_state=42
)

# model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# model training 
model.fit(X_train, y_train)

# accuracy score 
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# title 

st.title("💪 BODY TYPE PREDICTION ")

st.write(
    "Predict whether you are Ectomorph, Mesomorph or Endomorph using Machine Learning"
)

st.write("---")


# image
st.image(
    r"C:\Users\Nadkarni Omkar\OneDrive\Desktop\bodytype_prediction\fitness.jpg",
    use_container_width=True
)

st.write("")

# input section 
col1, col2 = st.columns(2)

with col1:

    st.subheader("📏 Physical Details")

    height = st.number_input(
        "Enter Height (cm)",
        min_value=100.0,
        max_value=250.0,
        value=175.0
    )

    weight = st.number_input(
        "Enter Weight (kg)",
        min_value=20.0,
        max_value=200.0,
        value=70.0
    )

with col2:

    st.subheader("🔥 Fitness Details")

    bmi = st.number_input(
        "Enter BMI",
        min_value=10.0,
        max_value=50.0,
        value=22.0
    )

    bodyfat = st.number_input(
        "Enter Body Fat %",
        min_value=1.0,
        max_value=60.0,
        value=15.0
    )

st.write("")


# prediction

if st.button("🚀 Predict Body Type"):

    # USER DATA
    new_person = [[height, weight, bmi, bodyfat]]

    # PREDICTION
    prediction = model.predict(new_person)

    result = prediction[0]

    # RESULT BOX
    st.markdown(
        f"""
        <div class="result-box">
            Predicted Body Type: {result}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # BODY TYPE MESSAGES

    if result == "Ectomorph":

        st.info("⚡ Lean and slim body type with fast metabolism.")

        st.image(
            r"C:\Users\Nadkarni Omkar\OneDrive\Desktop\bodytype_prediction\ectomorph.jpg",
            use_container_width=True
        )

    elif result == "Mesomorph":

        st.success("💪 Athletic and muscular body type.")

        st.image(
            r"C:\Users\Nadkarni Omkar\OneDrive\Desktop\bodytype_prediction\mesomorph.webp",
            use_container_width=True
        )

    else:

        st.warning("🔥 Broad body structure with higher fat storage.")

        st.image(
            r"C:\Users\Nadkarni Omkar\OneDrive\Desktop\bodytype_prediction\endomorph.webp",
            use_container_width=True
        )


# SIDEBAR
st.sidebar.title("📊 Model Information")

st.sidebar.success(f"Accuracy: {round(accuracy * 100, 2)}%")

st.sidebar.write("Algorithm Used:")
st.sidebar.write("Random Forest Classifier")

st.sidebar.write("Features Used:")
st.sidebar.write("- Height")
st.sidebar.write("- Weight")
st.sidebar.write("- BMI")
st.sidebar.write("- Body Fat %")

st.sidebar.info("Machine Learning Mini Project")