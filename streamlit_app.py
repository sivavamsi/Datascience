import streamlit as st
import requests
import os
import pandas as pd
from resume_analyzer import final_df_preparation  # Your matching logic

FASTAPI_URL = "http://localhost:8000"
JD_DIR = os.path.abspath("job_descriptions")
RESUME_BASE_DIR = os.path.abspath("resumes")

os.makedirs(JD_DIR, exist_ok=True)
os.makedirs(RESUME_BASE_DIR, exist_ok=True)

st.title("Resume Matcher App")

# 1. Upload JD
st.header("Upload Job Description")
jd_file = st.file_uploader("Upload JD", type=["txt", "pdf", "docx"], key="jd_upload")
if jd_file:
    response = requests.post(
        f"{FASTAPI_URL}/upload/jd/",
        files={"file": (jd_file.name, jd_file, jd_file.type)}
    )
    st.success(f"JD uploaded: {jd_file.name}")

# 2. Select Position from existing folders
st.header("Select Job Position")
existing_positions = [
    f for f in os.listdir(RESUME_BASE_DIR)
    if os.path.isdir(os.path.join(RESUME_BASE_DIR, f))
]

selected_position = st.selectbox("Choose from existing positions", existing_positions) if existing_positions else None

# 3. Upload Resumes
st.header("Upload Resumes")
resume_files = st.file_uploader(f"Upload resumes for {selected_position}", type=["txt", "pdf", "docx"], accept_multiple_files=True)

if st.button("Upload Resumes"):
    if resume_files and selected_position:
        for resume_file in resume_files:
            files = {"file": (resume_file.name, resume_file, resume_file.type)}
            data = {"position": selected_position}
            res = requests.post(f"{FASTAPI_URL}/upload/resume/", files=files, data=data)
            st.success(f"Uploaded: {resume_file.name} → {res.json().get('saved_to')}")
    else:
        st.warning("Please select a position and upload at least one file.")

# 4. Select JD file
st.header("Select JD to Match")
jd_list = os.listdir(JD_DIR)
selected_jd = st.selectbox("Select JD file", jd_list) if jd_list else None

# 5. Run Matcher
if st.button("Run Matching"):
    if selected_position and selected_jd:
        resume_path = os.path.join(RESUME_BASE_DIR, selected_position)
        jd_path = os.path.join(JD_DIR, selected_jd)
        job_id = selected_jd.rsplit('.', 1)[0]
        # Make sure your `final_df_preparation` uses these two paths
        df = final_df_preparation(job_id=job_id,position=selected_position)

        st.dataframe(df)
    else:
        st.warning("⚠️ Please select both a job position and a JD file to run matching.")
