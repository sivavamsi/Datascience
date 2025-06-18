def job_profile_template(job_description):
    prompt = f"""
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

    return prompt

def resume_template(resume):
    prompt = f"""
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
    return prompt