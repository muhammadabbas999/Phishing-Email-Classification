
import streamlit as st
import requests

# Page Configuration
st.set_page_config(
    page_title="Phishing Email Detector",
    page_icon="🛡️",
    layout="wide"
)

# Dashboard Heading
st.title("🛡️ Phishing Email Risk Classification System")
st.write("AI-powered system for detecting phishing emails.")

st.divider()

# Email Input
st.subheader("📧 Email Security Scanner")

email_text = st.text_area(
    "Enter Email Subject and Body:",
    height=200,
    placeholder="Paste your email content here..."
)

# Analyze Button
if st.button("Analyze Email", type="primary"):

    if not email_text.strip():
        st.warning("Please enter email text first.")

    else:
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"email_text": email_text},
                timeout=30
            )

            if response.status_code == 200:

                result = response.json()

                st.divider()
                st.subheader("📊 Email Security Report")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Prediction",
                        result["prediction"]
                    )

                with col2:
                    st.metric(
                        "Risk Level",
                        result["risk_level"]
                    )

                with col3:
                    st.metric(
                        "Phishing Probability",
                        f'{result["phishing_probability"]}%'
                    )

                # Result Message
                if result["prediction"] == "Phishing":
                    st.error("⚠️ Warning! This email may be a phishing attempt.")
                else:
                    st.success("✅ This email is classified as legitimate.")

                # Suspicious Indicators
                st.subheader("🔍 Explainable AI - Suspicious Indicators")

                indicators = result["suspicious_indicators"]

                if indicators:
                    for item in indicators:
                        st.write(
                            f'⚠️ **{item["word"]}** '
                            f'(Contribution: {item["contribution"]})'
                        )
                else:
                    st.info("No suspicious indicators identified by the model.")

            else:
                st.error(f"API Error: {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("Backend API is not running. Please start FastAPI first.")

        except requests.exceptions.RequestException as error:
            st.error(f"Request failed: {error}")
