from langchain.prompts import PromptTemplate

from langchain.prompts import PromptTemplate

job_template = PromptTemplate.from_template("""
You are an intelligent parser that extracts structured data from job descriptions.

Your task is to extract the following:
- skillset: list of all relevant skills and tools mentioned
- min_experience: minimum years of experience required (use -1 if not mentioned)
- max_experience: maximum years of experience required (use -1 if not mentioned)
- job_loacation: list of locations mentioned that should be a town or city. should not include the country

🧾 Job Description:
{job_description}

📤 Output (respond with ONLY valid JSON in this format):

{{
  "skillset": ["<list_of_skills>"],
  "min_experience": <minimum_experience_in_years_or_-1>,
  "max_experience": <maximum_experience_in_years_or_-1>,
  "job_loacation": ["<list_of_locations>"]
}}

📌 Rules:
- If experience years are mentioned as a range (e.g., "5-7 years"), extract both min and max.
- If only one experience number is mentioned (e.g., "5+ years"), set min to that number and max to -1.
- Keep keys  exactly as shown above.Do not add any keys.
- Do not add any text or explanation outside the JSON.
- Final answer should be the your Observation: json output
""")


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
- Final answer should be the Observation: json output
Now parse the resume and return only the JSON.
"""
)