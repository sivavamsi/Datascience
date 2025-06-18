from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

app = FastAPI()

# Allow Streamlit to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR_JD = os.path.abspath("job_descriptions")
UPLOAD_DIR_RESUME = os.path.abspath("resumes")
os.makedirs(UPLOAD_DIR_JD, exist_ok=True)
os.makedirs(UPLOAD_DIR_RESUME, exist_ok=True)

@app.post("/upload/jd/")
async def upload_jd(file: UploadFile = File(...)):
    path = os.path.join(UPLOAD_DIR_JD, file.filename)
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename}

@app.post("/upload/resume/")
async def upload_resume(position: str = Form(...), file: UploadFile = File(...)):
    position_dir = os.path.join(UPLOAD_DIR_RESUME, position)
    os.makedirs(position_dir, exist_ok=True)

    path = os.path.join(position_dir, file.filename)
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"saved_to": path}
