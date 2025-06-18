import asyncio
from langchain_community.document_loaders import PyPDFLoader
import google.generativeai as genai
from dotenv import load_dotenv
import os
import ast
import sys
import statistics
load_dotenv()
from google.genai import types
import pandas as pd
from chromadb import Documents, EmbeddingFunction, Embeddings
import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI
from template import job_template,resume_template
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.chains.llm import LLMChain
import json
from template_format import job_profile_template, resume_template
google_api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0,google_api_key =google_api_key)
global client
client = genai.configure(api_key=google_api_key)
chroma_client = chromadb.PersistentClient(path = './chroma')


def skill_match_percentage(candidate_skills, jd_skills_lower):
    if not jd_skills_lower:
        return 0
    candidate_skills_lower = set(skill.lower() for skill in candidate_skills)
    match_count = len(candidate_skills_lower & jd_skills_lower)
    return (match_count / len(jd_skills_lower)) * 100

def location_match(x,job_location):
        x = x.lower()

        job_location = [i.lower() for i in job_location]
        if x in job_location:
            return 1
        else:
            return 0




def stability_score(x):
    try:
        data = list(ast.literal_eval(x).values())
    except:
        data = list(x.values())
    if len(data)==1 and data[0]<2:
        stable = -1
    elif len(data)==1 and data[0]>=2:
        stable=data[0]-2
    else:

        stable = statistics.stdev(data)
    return stable

def load_json(path, name):
    file_path = os.path.join(path, f"{name}.json")
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[Error] File not found: {file_path}")
    except json.JSONDecodeError as e:
        print(f"[Error] Failed to decode JSON in {file_path}: {e}")
    except PermissionError:
        print(f"[Error] Permission denied while accessing: {file_path}")
    except Exception as e:
        print(f"[Error] Unexpected error while loading JSON from {file_path}: {e}")
    return []
def save_json(parsed_resumes, path, name):
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"{name}.json")
    with open(file_path, "w") as f:
        json.dump(parsed_resumes, f, indent=2)
    print(f"Saved to: {file_path}")

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        EMBEDDING_MODEL_ID = "models/gemini-embedding-exp-03-07"  # @param ["models/embedding-001", "models/text-embedding-004", "models/gemini-embedding-exp-03-07", "models/gemini-embedding-exp"] {"allow-input": true, "isTemplate": true}
        title = "Custom query"
        response = client.models.embed_content(
            model=EMBEDDING_MODEL_ID,
            contents=input,
            config=types.EmbedContentConfig(
                task_type="retrieval_document",
                title=title
            )
        )

        return response.embeddings[0].values


def get_db(collection_name):
    try:
        collection_list = [i.name for i in chroma_client.list_collections()]
        print(f"list of db's in vector store{collection_list}")
        if collection_name not in collection_list:
            chroma_client.create_collection(name=collection_name, embedding_function=GeminiEmbeddingFunction())
            db = chroma_client.get_collection(collection_name)
            print(f"Created the New Collection : {collection_name}")
        else:
            db = chroma_client.get_collection(collection_name)
            print(f"Collection Already exits : {collection_name}")
            print(f"Total records : {db.count()} , Database id : {db.id} and Database Config : {db.configuration}")

        return db

    except Exception as e:
        print("connecting db error:", e)


def inserting(extracted_text, name_id,db):
    try:
        if name_id not in db.get()['ids']:
            db.add(documents=extracted_text, ids=name_id) #addmetadatas=[{"source": "dataset"}]
            print(f"Added the content : {name_id}")
            print(f"Total records : {db.count()}")
        else:
            print(f"Same Ids Already Exits in DB : {name_id} Skiping this content")
    except Exception as e:
        print("While inserting data to vector store error:", e)
def extract_ner(prompt):
    token = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=token)
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    print(response.text)
    return response.text


def get_pdf_files(directory):
    return [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if file.endswith(".pdf")
    ]

# Example usage


async def extract_pages(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    return pages

def sync_vector_db(subject,tag):
    resume_dir_path = os.path.abspath(os.path.join(tag, subject))
    pdf_resumes = get_pdf_files(resume_dir_path)
    if subject == "":
        collection_name = tag
    else:
        collection_name = subject + "_" + tag
    resume_db  = get_db(collection_name=collection_name)
    print(resume_db)
    for resume_path in pdf_resumes:
        print(f"\n📄 File: {resume_path}")
        resume_name = os.path.splitext(os.path.basename(resume_path))[0]
        print(resume_name)
        pages = asyncio.run(extract_pages(resume_path))
        inserting(extracted_text=pages[0].page_content, name_id=resume_name,db = resume_db)
    print("*"*23)
    return resume_db
def similarity_check(resume_collection,jobdesc_collection,job_id):
    job_data = jobdesc_collection.get(ids=[job_id], include=["embeddings", "documents"])
    job_embedding = job_data["embeddings"][0]
    results = resume_collection.query(
        query_embeddings=[job_embedding],
        n_results=10,
    )
    df = pd.DataFrame({
        'ID': results['ids'][0],
        'Text': results['documents'][0],
        'Distance': results['distances'][0]
    })
    df = df.sort_values(by='Distance')
    return df

def resume_excutor(resume_collection, position):
    all_resume_ids = resume_collection.get()['ids']
    resume_ids = load_json("./parsed_data", position+"_resumes")
    resume_list = [i["resume_id"] for i in resume_ids]
    parsed_resumes = []
    for resume_id  in all_resume_ids:
        if resume_id not in resume_list:
            resume_text = resume_collection.get(ids=[resume_id])["documents"][0]

            resume_output = extract_ner(prompt=resume_template(resume_text))
            # resume_output = agent_executor.run(
            #     "You are an expert resume parser. Parse the following resume and return **only** a JSON dictionary in this exact format: " + resume_text)
            lines = resume_output.strip().splitlines()
            cleaned_str = "\n".join(lines[1:-1])
            parsed_dict = json.loads(cleaned_str)
            parsed_resumes.append({
                "resume_id": resume_id,
                "parsed_output": parsed_dict
            })
            print(f"Processed:{resume_id}")


        else:
            print(f"{resume_id} is already processed. skipping...")
    parsed_resumes.extend(resume_ids)
    if parsed_resumes!=[]:
        save_json(parsed_resumes, "./parsed_data", position+"_resumes")
def jd_excutor(jobdesc_collection, position,job_id):
    #all_jd_ids = jobdesc_collection.get()['ids']
    jd_ids = load_json("./parsed_data", job_id+"_jd")
    if jd_ids!=[]:
        jd_id_id = jd_ids["job_id"]
    else:
        jd_id_id=""
    parsed_dict={}
    if job_id!=jd_id_id:
        job_text = jobdesc_collection.get(ids=[job_id])["documents"][0]
        #job_profile_template
        job_output = extract_ner(prompt=job_profile_template(job_text))
        #job_output = agent_executor.run("You are an expert job description parser.Parse the following job description and extract the following information in JSON dictionary format only — no explanations, just the JSON output: " + job_text)
        lines = job_output.strip().splitlines()
        cleaned_str = "\n".join(lines[1:-1])
        parsed_dict = json.loads(cleaned_str)
        parsed_dict.update({"job_id": job_id})
        #parsed_dict = [parsed_dict]
        #jd_ids.append(parsed_dict)

        print(f"Processed:{job_id}")

    else:
        print(f"{job_id} is already processed. skipping...")
    if parsed_dict!={}:
        save_json(parsed_dict, "./parsed_data", job_id+"_jd")
####################################################################
#position="DataScientist"
# resume_collection=sync_vector_db(subject = position,tag="resumes")
# jobdesc_collection=sync_vector_db(subject = "",tag="job_descriptions")
#

# job_chain = LLMChain(llm=llm, prompt=job_template)
# resume_chain = LLMChain(llm=llm, prompt=resume_template)
#
# tools = [
#     Tool(
#         name="JobDescriptionParser",
#         func=lambda job_description: job_chain.run(job_description=job_description),
#         description="Parses job description and returns structured JSON output"
#     ),
#     Tool(
#         name="ResumeParser",
#         func=lambda resume: resume_chain.run(resume=resume),
#         description="Parses resume and returns structured JSON output"
#     )
# ]
#
# agent_executor = initialize_agent(
#     tools,
#     llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True,
# )



def final_df_preparation(job_id,position):

    resume_collection = sync_vector_db(subject=position, tag="resumes")
    jobdesc_collection = sync_vector_db(subject="", tag="job_descriptions")

    resume_excutor(resume_collection, position)
    jd_excutor(jobdesc_collection, position, job_id)
    similarity_df = similarity_check(resume_collection, jobdesc_collection, job_id)
    resume_data = load_json("./parsed_data", position + "_resumes")
    jd_data = load_json("./parsed_data", job_id + "_jd")

    df = pd.DataFrame(resume_data)
    expanded_df = df['parsed_output'].apply(pd.Series)
    df = pd.concat([df.drop(columns=['parsed_output']), expanded_df], axis=1)
    df = df.merge(similarity_df,how="inner",left_on="resume_id",right_on="ID")
    jd_skills_lower = set(skill.lower() for skill in jd_data['skillset'])
    df['Skill_Match_%'] = df['Skillset'].apply(lambda x: skill_match_percentage(x, jd_skills_lower))
    #df['location_match'] = df["User_Location"].apply(location_match)
    df['location_match'] = df["User_Location"].apply(lambda x: location_match(x,jd_data["job_loacation"] ))
    df['stability_score'] = df['Canditate_Switches'].apply(stability_score)
    df_ranked = df.sort_values(
        by=['Skill_Match_%', 'Distance', 'stability_score', 'location_match'],
        ascending=[False, True, False, False]
    ).reset_index(drop=True)

    df_ranked['Rank'] = df_ranked.index + 1
    return df_ranked
job_id = "JD for Machine Learning Profile"
position = "DataScientist"
print(final_df_preparation(job_id,position))
