import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from evaluator import RobustJobMismatchEvaluator
import plotly.graph_objects as go

# ----------------- Custom CSS -----------------
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom right, #f0f7ff, #ffffff);
        font-family: 'Segoe UI', sans-serif;
    }
    .stTextInput > label, .stTextArea > label, .stSelectbox > label {
        font-weight: 600;
    }
    .result-box {
        background-color: #e6f2ff;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #d0e6ff;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- Load Model & Evaluator -----------------
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = DistilBertTokenizer.from_pretrained("Model_folder", local_files_only=True)
    model = DistilBertForSequenceClassification.from_pretrained("Model_folder", local_files_only=True)
    model.eval()
    return model, tokenizer

@st.cache_resource
def load_evaluator():
    return RobustJobMismatchEvaluator()

model, tokenizer = load_model_and_tokenizer()
evaluator = load_evaluator()

# ----------------- Gauge Drawing Function -----------------
def draw_gauge(title, value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "royalblue"},
            'bgcolor': "white",
            'borderwidth': 1,
            'steps': [
                {'range': [0, 50], 'color': '#f8d7da'},
                {'range': [50, 75], 'color': '#fff3cd'},
                {'range': [75, 100], 'color': '#d4edda'}
            ],
        }
    ))
    fig.update_layout(height=250, margin=dict(t=20, b=0, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)

# ----------------- Initialize Session -----------------
if "show_results" not in st.session_state:
    st.session_state.show_results = False

# ----------------- UI -----------------
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🕵️ Job Fraud & Mismatch Evaluator</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)

if not st.session_state.show_results:
    # ------------- Input Form -------------
    with st.form("job_form"):
        st.markdown("### 📌 Required Job Details")

        col1, col2 = st.columns(2)
        with col1:
            job_title = st.text_input("📝 Job Title *", placeholder="e.g., Data Analyst")
        with col2:
            employment_type = st.selectbox("👔 Employment Type *", [
                "Full-time", "Internship", "Internship & Graduate", "Other Part-time", "Temporary"
            ])

        job_description = st.text_area("📄 Job Description *", height=150)
        skill_desc = st.text_area("🛠️ Skills Required *", height=100)
        location = st.text_input("📍 Location *", placeholder="e.g., Bangalore, India")

        st.markdown("### 📝 Optional Details")
        salary_range = st.text_input("💰 Salary Range", placeholder="e.g., 4-6 LPA")
        industry = st.text_input("🏢 Industry", placeholder="e.g., IT Services")
        company_profile = st.text_area("🏙️ Company Profile", height=100)

        submitted = st.form_submit_button("🚀 Evaluate", use_container_width=True)

    if submitted:
        if not all([job_title.strip(), job_description.strip(), skill_desc.strip(), location.strip(), employment_type.strip()]):
            st.error("⚠️ Please fill all the required fields marked with *.")
        else:
            # Store data in session
            st.session_state.job_inputs = {
                "Job Title": job_title,
                "Employment Type": employment_type,
                "Job Description": job_description,
                "Skills Required": skill_desc,
                "Location": location,
                "Salary Range": salary_range,
                "Industry": industry,
                "Company Profile": company_profile,
            }
            st.session_state.show_results = True
            st.experimental_set_query_params(page="results")

# ----------------- Results Page -----------------
if st.session_state.show_results:
    st.markdown("## 📊 Prediction Results")

    inputs = st.session_state.get("job_inputs", {})

    # Combine all relevant text for model prediction
    combined_text = f"{inputs['Job Title']} {inputs['Job Description']} {inputs['Skills Required']} {inputs['Employment Type']} {inputs['Location']}"

    # Model Prediction
    temperature = 2.0
    tokens = tokenizer(combined_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**tokens)
        probs = torch.softmax(outputs.logits / temperature, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item() * 100
        class_labels = ["✅ Real Job Posting", "🚨 Fake Job Posting"]
        prediction_label = class_labels[pred_idx]

    # Mismatch Score - pass lists of job titles and descriptions
    mismatch_score = evaluator.evaluate([inputs["Job Title"]], [inputs["Job Description"]])

    col1, col2 = st.columns(2)
    with col1:
        draw_gauge("Model Confidence", round(confidence, 2))
    with col2:
        draw_gauge("Job-Role Match Score", round(mismatch_score, 2))

    st.markdown(f"### 🏷️ Prediction: **{prediction_label}**")

    # Job Summary
    with st.expander("📄 Job Summary (Your Input)", expanded=True):
        for key, val in inputs.items():
            if val.strip():
                st.markdown(f"**{key}:** {val}")

    # Back button to submit again
    if st.button("🔄 Evaluate Another Job"):
        st.session_state.show_results = False
        st.experimental_set_query_params()
