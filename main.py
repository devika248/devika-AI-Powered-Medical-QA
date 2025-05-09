# main.py

import os
import toml
import types
import pandas as pd
from botocore.client import Config
import ibm_boto3
import streamlit as st
from streamlit.runtime.caching import cache_resource
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models import Model
from sklearn.model_selection import train_test_split

# --- Streamlit Page Config ---
st.set_page_config(page_title="Medical QA with WatsonX", layout="centered")
st.title("🩺 Medical Question Answering App")

# --- Secrets Configuration ---
secrets = {
    "API_KEY": "_na8QIUga8x-gNsHoDyPRDFL8NgvPtiWSrMJkyr2em7a",
    "PROJECT_ID": "94118a53-48f5-4ade-8062-540463da4699",
    "URL": "https://eu-de.ml.cloud.ibm.com"
}
secrets_dir = os.path.expanduser("~/.streamlit")
os.makedirs(secrets_dir, exist_ok=True)
secret_file_path = os.path.join(secrets_dir, "secrets.toml")
with open(secret_file_path, 'w') as f:
    toml.dump(secrets, f)
os.environ['STREAMLIT_SECRETS_FILE'] = secret_file_path

# --- Load Model ---
@cache_resource(show_spinner=False)
def load_model():
    parameters = {
        GenParams.DECODING_METHOD: "greedy",
        GenParams.RANDOM_SEED: 33,
        GenParams.REPETITION_PENALTY: 1.0,
        GenParams.MIN_NEW_TOKENS: 1,
        GenParams.MAX_NEW_TOKENS: 100
    }
    model = Model(
        model_id=ModelTypes.FLAN_T5_XXL,
        credentials={"apikey": secrets["API_KEY"], "url": secrets["URL"]},
        project_id=secrets["PROJECT_ID"],
        params=parameters
    )
    return model

model = load_model()

# --- Load MedQuAD Data from local CSV file ---
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('MedQuad.csv')[['Question', 'Answer']].dropna().reset_index(drop=True)
        return data
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return pd.DataFrame(columns=["Question", "Answer"])

data = load_data()

# --- User Question Input ---
st.subheader("Ask a Medical Question")
question = st.text_area("Enter your question:", "What is asthma")

if st.button("Get Answer") and question:
    prompt = f"Q: {question}\nA:"
    with st.spinner("Generating answer with WatsonX..."):
        response = model.generate_text(prompt=prompt)
        answer = response.get("generated_text", str(response)) if isinstance(response, dict) else str(response)
    st.success("Answer:")
    st.write(answer)

# --- Show Sample Data ---
if st.checkbox("Show Sample MedQuAD Questions"):
    st.subheader("Sample Medical Questions")
    st.dataframe(data.head(10))

# --- Example Q-A Inference with Ground Truth ---
st.subheader("🔍 Example from MedQuAD Test Set")

if not data.empty:
    questions = data['Question']
    answers = data['Answer']
    q_train, q_test, a_train, a_test = train_test_split(questions, answers, test_size=0.3, random_state=33)

    sample_index = 0  # you can make this a random index if desired
    sample_question = q_test.iloc[sample_index]
    sample_answer = a_test.iloc[sample_index]
    prompt = f"Q: {sample_question}\nA:"
    st.markdown(f"**Question:** {sample_question}")

    with st.spinner("Generating model answer..."):
        response = model.generate_text(prompt=prompt, params={"decoding_method": "greedy"})
        predicted = response.get("generated_text", str(response)) if isinstance(response, dict) else str(response)

    st.markdown("**🧠 Predicted Answer:**")
    st.write(predicted)

    st.markdown("**✅ Ground Truth:**")
    st.write(sample_answer)
else:
    st.warning("No data available to display an example.")