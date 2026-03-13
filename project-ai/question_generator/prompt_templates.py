"""
prompt_templates.py — Prompt Templates for Question Generator
==============================================================
This module contains all prompt strings used by question_generator.py
to generate interview questions via the Groq LLM API.

Keeping prompts separate from logic makes it easy to:
    - Tweak question style without touching generator logic
    - Add new question types in the future
    - Review and improve prompts independently

OPTIMIZATION NOTE:
    Previously used 4 separate prompts (technical, behavioral, hr, resume-based)
    requiring 4 separate Groq API calls. Now merged into a single prompt
    that returns all 4 question types in one JSON response — saving 3 API calls
    per session.
"""


def get_all_questions_prompt(
    skills: list,
    resume_text: str,
    job_description: str,
    num_questions: int = 5
) -> str:
    """
    Returns a single combined prompt that asks Groq to generate
    all 4 types of interview questions in one API call.

    This replaces the 4 separate prompts and reduces Groq API calls
    from 4 to 1 per question generation session.

    Parameters:
        skills (list):          List of skills extracted from the resume.
                                Example: ["python", "docker", "react"]
        resume_text (str):      Full raw text of the candidate's resume.
        job_description (str):  The full job description text.
        num_questions (int):    Number of questions per type. Default is 5.

    Returns:
        str: A formatted prompt string for the Groq API.

    Expected LLM response format (JSON):
        {
            "technical":    ["Q1?", "Q2?", "Q3?", "Q4?", "Q5?"],
            "behavioral":   ["Q1?", "Q2?", "Q3?", "Q4?", "Q5?"],
            "hr_general":   ["Q1?", "Q2?", "Q3?", "Q4?", "Q5?"],
            "resume_based": ["Q1?", "Q2?", "Q3?", "Q4?", "Q5?"]
        }
    """
    skills_str = ", ".join(skills) if skills else "general software development"

    return f"""You are an expert interviewer preparing a complete interview question set.

CANDIDATE SKILLS: {skills_str}

CANDIDATE RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}

Generate exactly {num_questions} questions for each of the following 4 types:

1. technical — Questions specific to the candidate's skills and the job requirements.
   Include questions on distributed systems, system design tradeoffs, performance bottlenecks,
   observability, and OS/networking concepts where relevant to the job description.
   Mix conceptual and practical. Range from intermediate to senior-level difficulty.

2. behavioral — STAR-format questions (Situation, Task, Action, Result).
   Start with "Tell me about a time..." or "Describe a situation..." or "Give me an example..."
   Focus on teamwork, problem-solving, leadership, conflict resolution.

3. hr_general — HR and general questions about motivation, career goals,
   strengths, weaknesses, culture fit, and why this company/role.

4. resume_based — Personalized questions based SPECIFICALLY on the candidate's resume.
   Ask about specific projects, experiences, transitions, or achievements mentioned.
   Questions should feel personalized, not generic.

Rules:
- Do NOT include answers
- Each type must have exactly {num_questions} questions
- Questions must be relevant to the job description

Respond ONLY with a valid JSON object. No explanation, no markdown, no extra text.
The JSON must have exactly this structure:
{{
    "technical":    ["Q1?", "Q2?", ...(exactly {num_questions} items)],
    "behavioral":   ["Q1?", "Q2?", ...(exactly {num_questions} items)],
    "hr_general":   ["Q1?", "Q2?", ...(exactly {num_questions} items)],
    "resume_based": ["Q1?", "Q2?", ...(exactly {num_questions} items)]
}}
"""