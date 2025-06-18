from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()

job_template = PromptTemplate(
    input_variables=["job_description"],
    template="""
You are an expert job description parser. Given the job description content below, extract the following structured information in valid JSON format:

Input:
{{"job_profile": "{job_description}"}}

Output:
{{
  "skillset": ["<list_of_skills>"],
  "min_experience": <minimum_experience_in_years_or_-1>,
  "max_experience": <maximum_experience_in_years_or_-1>,
  "job_loacation": ["<list_of_locations>"]
}}

Rules:
- "skillset" should include only technical or domain-specific skills mentioned in the job description (e.g., Python, Java, SQL).
- "min_experience" should reflect the minimum experience required, in years, from any context (technical or non-technical). Use -1 if not mentioned.
- "max_experience" should reflect the upper experience limit, in years, if mentioned. Use -1 if not provided.
- "job_loacation" should include all cities or locations mentioned in the job description.
- All keys must be present in the output. If any value is missing in the description, return an empty list or -1 accordingly.

Now parse the job description and return only the JSON.
"""
)

resume_template = PromptTemplate(
    input_variables=["resume"],
    template="""
You are an expert resume parser. Given the resume content below, extract the following structured information in valid JSON format:

Input:
{{"resume_content": "{resume}"}}

Output:
{{
  "Candidate_Name": "<full_name>",
  "Email_ID": "<email_address>",
  "Phone_Number": "<phone_number>",
  "User_Location": "<city_or_region>",
  "Skillset": ["<list_of_skills>"],
  "Total_IT_Experience": <total_years_in_digits>,
  "Canditate_Switches": {{
    "<Company_Name_1>": <years_of_experience_in_digits>,
    "<Company_Name_2>": <years_of_experience_in_digits>
  }}
}}

Rules:
- "Candidate_Name" should be the full name of the candidate, if available.
- "Email_ID" should be a valid email address found in the resume.
- "Phone_Number" should include the country code if available, or just a 10-digit number.
- "Skillset" should include only technical or professional skills (e.g., Python, Java, Docker).
- "User_Location" should be the most recent or clearly mentioned location.
- "Total_IT_Experience" should be a whole number (in years) calculated from the experience descriptions.
- "Canditate_Switches" should include each company mentioned in the resume with corresponding years of experience there.
- If information is missing, leave it blank (e.g., "" or 0).

Now parse the resume and return only the JSON.
"""
)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0,google_api_key =os.getenv("GOOGLE_API_KEY"))
print(llm)
job_chain = LLMChain(llm=llm, prompt=job_template)
resume_chain = LLMChain(llm=llm, prompt=resume_template)

tools = [
    Tool(
        name="JobDescriptionParser",
        func=lambda job_description: job_chain.run(job_description=job_description),
        description="Parses job description and returns structured JSON output"
    ),
    Tool(
        name="ResumeParser",
        func=lambda resume: resume_chain.run(resume=resume),
        description="Parses resume and returns structured JSON output"
    )
]

agent_executor = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

job_text = job_collection.get(ids=["job001"])["documents"][0]
resume_text = resume_collection.get(ids=["resume001"])["documents"][0]

# Extract JSON from job and resume
job_output = agent_executor.run("Parse this job description: " + job_text)
resume_output = agent_executor.run("Parse this resume: " + resume_text)

print("Parsed Job JSON:\n", job_output)
print("Parsed Resume JSON:\n", resume_output)


from langgraph.graph import StateGraph

graph = StateGraph()

graph.add_node("load_job", load_job_node)
graph.add_node("embed_job", embed_job_node)
graph.add_node("search_chroma", chroma_search_node)
graph.add_node("score_resumes", score_resumes_node)
graph.set_entry_point("load_job")
graph.set_output("score_resumes")

workflow = graph.compile()
results = workflow.invoke(input={"job_id": "12345"})