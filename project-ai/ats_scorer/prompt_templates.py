"""
prompt_templates.py — Prompt Templates for ATS Scorer
======================================================
This module contains all prompt strings used by ats_scorer.py
to perform LLM-based semantic scoring via the Groq API.

Keeping prompts separate from logic makes it easy to:
    - Tweak scoring criteria without touching scorer logic
    - Add new scoring categories in the future
    - Review and improve prompts independently

OPTIMIZATION NOTE:
    Previously used 3 separate prompts (semantic, experience, education)
    requiring 3 separate Groq API calls. Now merged into a single prompt
    that returns all 3 scores in one JSON response — saving 2 API calls
    per session.
"""


def get_combined_llm_scores_prompt(
    resume_text: str,
    experience_text: str,
    education_text: str,
    job_description: str
) -> str:
    """
    Returns a single combined prompt that asks Groq to evaluate
    semantic match, experience match, and education match all at once.

    This replaces the 3 separate prompts (get_semantic_match_prompt,
    get_experience_match_prompt, get_education_match_prompt) and reduces
    Groq API calls from 3 to 1 per ATS scoring session.

    Parameters:
        resume_text (str):      Full raw text of the candidate's resume.
        experience_text (str):  Work experience section from the resume.
        education_text (str):   Education section from the resume.
        job_description (str):  Full text of the job description.

    Returns:
        str: A formatted prompt string for the Groq API.

    Expected LLM response format (JSON):
        {
            "semantic_match": {
                "score": 78,
                "reason": "Strong backend experience but lacks cloud skills."
            },
            "experience_match": {
                "score": 70,
                "reason": "2 years of relevant experience, role requires 3-5 years."
            },
            "education_match": {
                "score": 90,
                "reason": "B.Tech in Computer Science matches the requirement."
            }
        }
    """
    return f"""You are an expert ATS (Applicant Tracking System) evaluator and recruiter.

Evaluate the candidate's resume against the job description across three dimensions
and return all three scores in a single JSON response.

FULL RESUME:
{resume_text}

WORK EXPERIENCE:
{experience_text}

EDUCATION:
{education_text}

JOB DESCRIPTION:
{job_description}

Evaluate and score the following three dimensions:

1. semantic_match — How well does the overall resume match the job description?
   Consider: overall relevance, technical stack alignment, role suitability.

2. experience_match — How well does the work experience match the job requirements?
   Consider: years of experience, relevance of past roles, seniority level match.

3. education_match — How well does the education match the job requirements?
   Consider: degree level, field of study relevance, certifications.

Respond ONLY with a valid JSON object. No explanation, no markdown, no extra text.
The JSON must have exactly this structure:
{{
    "semantic_match":   {{"score": <0-100>, "reason": "<one or two sentences>"}},
    "experience_match": {{"score": <0-100>, "reason": "<one or two sentences>"}},
    "education_match":  {{"score": <0-100>, "reason": "<one or two sentences>"}}
}}
"""