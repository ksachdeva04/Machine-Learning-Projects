
import streamlit as st
import pickle
import pandas as pd


# ============================================================
# 1. LOAD MODEL AND SCALER
# ============================================================

@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)

    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    return model, scaler


model, scaler = load_artifacts()


# ============================================================
# 2. PAGE TITLE
# ============================================================

st.title("Customer Churn Prediction")

st.write(
    "Enter the customer's information below to estimate "
    "their likelihood of churning."
)


# ============================================================
# 3. CUSTOMER INPUTS
# ============================================================

st.header("Customer Information")

age = st.number_input(
    "Age", min_value=0, max_value=100, value=30
)

tenure = st.number_input(
    "Tenure", min_value=0, value=12
)

usage_frequency = st.number_input(
    "Usage Frequency", min_value=0, value=10
)

support_calls = st.number_input(
    "Support Calls", min_value=0, value=2
)

payment_delay = st.number_input(
    "Payment Delay", min_value=0, value=0
)

subscription_type = st.selectbox(
    "Subscription Type",
    ["Basic", "Standard", "Premium"]
)

contract_length = st.selectbox(
    "Contract Length",
    ["Monthly", "Quarterly", "Annual"]
)

total_spend = st.number_input(
    "Total Spend", min_value=0.0, value=1000.0
)

last_interaction = st.number_input(
    "Last Interaction", min_value=0, value=10
)

gender = st.selectbox(
    "Gender", ["Female", "Male"]
)


# ============================================================
# 4. ENCODE CATEGORICAL VARIABLES
# ============================================================

subscription_mapping = {
    "Basic": 0,
    "Standard": 1,
    "Premium": 2
}

contract_mapping = {
    "Monthly": 0,
    "Quarterly": 1,
    "Annual": 2
}

subscription_type_encoded = subscription_mapping[
    subscription_type
]

contract_length_encoded = contract_mapping[
    contract_length
]

gender_female = 1 if gender == "Female" else 0
gender_male = 1 if gender == "Male" else 0


# ============================================================
# 5. CREATE INPUT DATAFRAME
# ============================================================

feature_columns = [
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Subscription Type",
    "Contract Length",
    "Total Spend",
    "Last Interaction",
    "Gender_Female",
    "Gender_Male"
]

input_data = pd.DataFrame([{
    "Age": age,
    "Tenure": tenure,
    "Usage Frequency": usage_frequency,
    "Support Calls": support_calls,
    "Payment Delay": payment_delay,
    "Subscription Type": subscription_type_encoded,
    "Contract Length": contract_length_encoded,
    "Total Spend": total_spend,
    "Last Interaction": last_interaction,
    "Gender_Female": gender_female,
    "Gender_Male": gender_male
}])

# Enforce the exact feature order used in training.
input_data = input_data[feature_columns]


# ============================================================
# 6. PREDICTION
# ============================================================

if st.button("Predict Churn", type="primary"):

    try:
        # Scale using the saved training scaler.
        input_scaled = scaler.transform(input_data)

        # Predict the class.
        prediction = model.predict(input_scaled)[0]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ Customer is likely to churn.")
        elif prediction == 0:
            st.success("✅ Customer is unlikely to churn.")
        else:
            st.warning(f"Predicted class: {prediction}")

        # Calculate probabilities using the actual class labels.
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                input_scaled
            )[0]

            classes = list(model.classes_)

            # Confirm the target classes are present.
            if 0 not in classes or 1 not in classes:
                st.warning(
                    "Expected target labels 0 (no churn) "
                    "and 1 (churn). Check your training labels."
                )
            else:
                no_churn_index = classes.index(0)
                churn_index = classes.index(1)

                no_churn_probability = float(
                    probabilities[no_churn_index]
                )

                churn_probability = float(
                    probabilities[churn_index]
                )

                st.subheader("Churn Probability")

                st.metric(
                    "Likelihood of Churning",
                    f"{churn_probability:.4%}"
                )

                st.progress(churn_probability)

                st.write(
                    f"**Churn:** "
                    f"{churn_probability:.4%}"
                )

                st.write(
                    f"**No churn:** "
                    f"{churn_probability:.4%}"
                )

                with st.expander("Technical model details"):
                    st.write("Model classes:", classes)
                    st.write(
                        "Raw probability output:",
                        probabilities.tolist()
                    )
                    st.write(
                        "Predicted class:",
                        prediction
                    )

        else:
            st.info(
                "This model does not support probability "
                "estimates through predict_proba()."
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
