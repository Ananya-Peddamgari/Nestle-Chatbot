# ================================================================
# Nestlé India AI Assistant - Professional NLP Chatbot with PDF Upload
# Author: Ananya Peddamgari | NLP Mini Project 2025
# ================================================================

import streamlit as st
import PyPDF2
import pandas as pd
import nltk
import re
import time
import os

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------
# Download necessary NLTK data (FIXED THE LookupError)
# ------------------------------
# Set a writable directory for NLTK data to ensure resources are found
NLTK_DATA_DIR = os.path.join(os.path.dirname(__file__), "nltk_data_cache")

# Add the local directory to the NLTK search path before checking/downloading
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_DIR)
    
if not os.path.exists(NLTK_DATA_DIR):
    os.makedirs(NLTK_DATA_DIR)

try:
    # Simply call download. NLTK will use the paths in nltk.data.path
    nltk.download("punkt", quiet=True) 
    nltk.download("stopwords", quiet=True)

except Exception as e:
    st.error(f"Failed to load NLTK data. Please check connection. Error: {e}")


# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Nesty – Nestlé India AI Assistant",
    page_icon="🍫",
    layout="wide",
)

# ------------------------------
# Custom CSS Styling
# ------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

* {
    font-family: 'Poppins', sans-serif;
}
body {
    background: linear-gradient(180deg, #ffffff 0%, #fafafa 60%, #f6eaea 100%);
}

/* HEADER */
.main-header {
    background: linear-gradient(90deg, #E41E26 0%, #0B4C8C 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 0 0 25px 25px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.25);
}
.main-header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0;
}
.main-header p {
    font-size: 1rem;
    margin-top: 0.4rem;
    opacity: 0.95;
}

}
.user-msg {
    background: #0B4C8C;
    color: white;
    padding: 12px 18px;
    border-radius: 20px 20px 0 20px;
    margin: 8px 0 2px auto;
    width: fit-content;
    max-width: 80%;
    font-size: 15px;
    animation: fadeIn 0.5s ease-in-out;
}
.bot-msg {
    background: #f7f7f7;
    color: #111;
    padding: 12px 18px;
    border-radius: 20px 20px 20px 0;
    margin: 2px 0 10px 0;
    width: fit-content;
    max-width: 80%;
    font-size: 15px;
    animation: fadeIn 0.5s ease-in-out;
}
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}

/* INPUT + BUTTON */
.stTextInput>div>div>input {
    border-radius: 10px;
    border: 1px solid #ccc;
    padding: 12px;
    font-size: 15px;
}
.stButton>button {
    background: #E41E26;
    color: white;
    font-weight: 600;
    border-radius: 10px;
    padding: 10px 28px;
    border: none;
    font-size: 15px;
}
.stButton>button:hover {
    background: #C41B1F;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: #f8f8f8;
    border-right: 1px solid #eee;
}
.sidebar-title {
    font-weight: 700;
    color: #E41E26;
    margin-bottom: 10px;
}
.sidebar-info {
    font-size: 14px;
    color: #333;
}

/* FOOTER */
.footer {
    margin-top: 25px;
    text-align: center;
    padding: 12px;
    color: #555;
    font-size: 13px;
    border-top: 1px solid #ddd;
}
.footer b {
    color: #E41E26;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Header Section
# ------------------------------
st.markdown("""
<div class="main-header">
    <h1>🤖 Nesty — Nestlé India AI Assistant</h1>
    <p>Explore insights, performance & initiatives directly from your uploaded Annual Report</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------
# Sidebar Section
# ------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/8/8e/Nestle_India_logo.png", width=150)
st.sidebar.markdown("""
<span class="sidebar-title">🏢 About Nestlé India</span>

<div class="sidebar-info">
Nestlé India is committed to enhancing quality of life and contributing to a healthier future.
It focuses on:
<ul>
<li>Nutrition, Health & Wellness</li>
<li>Sustainability & CSR</li>
<li>Innovation & Digitalization</li>
<li>Environmental Care</li>
<li>Creating Shared Value</li>
</ul>
</div>
---
<p style="font-size:13px;">💬 Chat powered by TF-IDF NLP model</p>
""", unsafe_allow_html=True)

# ------------------------------
# Helper Functions
# ------------------------------
def clean_text(txt):
    txt = txt.replace("\n", " ")
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()

def preprocess(txt):
    try:
        sw = set(stopwords.words("english"))
    except LookupError:
        # Fallback in case of persistent lookup issues
        sw = set() 

    txt = txt.lower()
    txt = re.sub(r"[^a-z\s]", " ", txt)
    return " ".join([w for w in txt.split() if w not in sw])

def pdf_to_text(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return clean_text(text)

@st.cache_data(show_spinner=False)
def build_tfidf_chunks(text, chunk_size=800):
    # This line now relies on the globally set NLTK paths
    sents = sent_tokenize(text) 
    chunks, cur = [], ""
    for s in sents:
        if len(cur) + len(s) <= chunk_size:
            cur += " " + s
        else:
            chunks.append(cur)
            cur = s
    chunks.append(cur)
    df = pd.DataFrame(chunks, columns=["chunk"])
    df["clean"] = df["chunk"].apply(preprocess)
    vec = TfidfVectorizer(max_features=3000)
    X = vec.fit_transform(df["clean"])
    return df, vec, X

def get_answer(q, df, vec, X):
    qv = vec.transform([preprocess(q)])
    sims = cosine_similarity(qv, X).flatten()
    top = sims.argsort()[-2:][::-1]
    return " ".join(df.iloc[top]["chunk"].values)

# ------------------------------
# Main Application
# ------------------------------
col1, col2, col3 = st.columns([1,3,1])
with col2:
    st.subheader("📂 Upload Nestlé India Annual Report (PDF)")
    
    pdf = st.file_uploader(
        "Upload Annual Report PDF",
        type=["pdf"],
        label_visibility="hidden"
    )

    if pdf:
        text = pdf_to_text(pdf)
        
        # Ensure the cache is cleared in deployment for this to work the first time after code change
        df, vec, X = build_tfidf_chunks(text) 
        
        st.success("✅ PDF loaded successfully! You can now chat below.")

        if "chat" not in st.session_state:
            st.session_state.chat = []

        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for who, msg in st.session_state.chat:
            if who == "You":
                st.markdown(f'<div class="user-msg">{msg}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-msg">{msg}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        q = st.text_input("💬 Ask something:", placeholder="e.g., What are Nestlé’s sustainability goals?")
        ask = st.button("Ask Nesty")

        if ask and q.strip():
            with st.spinner("Nesty is reading the report... 📖"):
                time.sleep(1)
                ans = get_answer(q, df, vec, X)
                if not ans.strip():
                    ans = "Sorry, I couldn’t find that information in the uploaded report."
                st.session_state.chat.append(("You", q))
                st.session_state.chat.append(("Nesty", ans))
            st.rerun()


    else:
        st.info("📎 Please upload the Annual Report PDF to start chatting with Nesty.")

# ------------------------------
# Footer Section
# ------------------------------
st.markdown("""
<div class="footer">
Made with ❤️ by <b>Ananya Peddamgari</b> | NLP Mini Project 2025  
</div>
""", unsafe_allow_html=True)
