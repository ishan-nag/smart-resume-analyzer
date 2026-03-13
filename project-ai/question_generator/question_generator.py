"""
question_generator.py — Interview Question Generator
=====================================================
This module generates interview questions for a candidate by sending
a single prompt to the Groq LLM API based on:
    - The candidate's parsed resume (output of resume_parser)
    - A job description provided by the recruiter/user

It generates 4 types of questions (5 each = 20 total) in ONE API call:
    1. Technical       — based on candidate skills + job description
    2. Behavioral      — STAR-format situational questions
    3. HR / General    — motivation, culture fit, career goals
    4. Resume-based    — personalized questions from resume content

OPTIMIZATION:
    Previously made 4 separate Groq API calls (one per question type).
    Now uses a single API call that returns all 4 types in one JSON response
    — reducing API calls from 4 to 1 per session.

Output format:
    A Python dictionary (easily serializable to JSON):
    {
        "technical":    [str, str, str, str, str],
        "behavioral":   [str, str, str, str, str],
        "hr_general":   [str, str, str, str, str],
        "resume_based": [str, str, str, str, str]
    }

For backend integration (Java Spring Boot):
    - Call generate_questions(parsed_resume, job_description)
    - Pass the dict from resume_parser directly as parsed_resume
    - Returns a dict → serialize with json.dumps() to send as JSON
    - Optionally call save_generated_questions(result, output_path) to persist

Dependencies:
    pip install groq python-dotenv
"""

import os
import json

# ── Import from shared/ ──
from shared.groq_client import get_groq_client, MODEL_CONFIGS
from shared.retry_handler import call_with_retry, parse_json_response

# ── Import combined prompt ──
try:
    from .prompt_templates import get_all_questions_prompt
except ImportError:
    from prompt_templates import get_all_questions_prompt


# ─────────────────────────────────────────────
# SECTION: Main Generate Function
# ─────────────────────────────────────────────

def generate_questions(parsed_resume: dict, job_description: str, num_questions: int = 5) -> dict:
    """
    Main function to generate all interview questions in a single API call.
    This is the PRIMARY function the backend should call.

    Previously made 4 separate Groq API calls (one per question type).
    Now makes 1 API call that returns all 4 question types at once.

    Parameters:
        parsed_resume (dict):
            The structured resume dictionary returned by resume_parser's
            parse_resume() function. Expected keys used:
                - "skills"   (list)  — for technical questions
                - "raw_text" (str)   — for resume-based questions
            Example:
                {
                    "name": "John Doe",
                    "skills": ["python", "docker", "react"],
                    "raw_text": "Full resume text...",
                    ...
                }

        job_description (str):
            The full text of the job description.
            Example: "We are looking for a backend engineer with Java Spring Boot..."

        num_questions (int):
            Number of questions to generate per type. Default is 5.
            Total questions = num_questions x 4 types = 20 by default.

    Returns:
        dict: A structured dictionary with 4 lists of questions.

        Example return value:
        {
            "technical":    ["What is the difference between REST and GraphQL?", ...],
            "behavioral":   ["Tell me about a time you handled a tight deadline...", ...],
            "hr_general":   ["Why are you interested in this role?", ...],
            "resume_based": ["I see you worked on a Spotify playlist generator...", ...]
        }

        On failure, returns:
        {
            "error": "Reason for failure"
        }

    Example usage (Python):
        from resume_parser import parse_resume
        from question_generator.question_generator import generate_questions

        resume    = parse_resume("data/sample_resume.pdf")
        job_desc  = open("data/sample_job_description.txt").read()
        questions = generate_questions(resume, job_desc)
        print(questions["technical"])
    """

    # ── Step 1: Validate inputs ──
    if not parsed_resume:
        return {"error": "parsed_resume is empty or None."}
    if not job_description or not job_description.strip():
        return {"error": "job_description is empty or None."}

    # ── Step 2: Extract fields from parsed resume ──
    skills   = parsed_resume.get("skills", [])
    raw_text = parsed_resume.get("raw_text", "")

    if not raw_text:
        raw_text = (
            f"Name: {parsed_resume.get('name', '')}\n"
            f"Skills: {', '.join(skills)}\n"
            f"Education: {parsed_resume.get('education', '')}\n"
            f"Experience: {parsed_resume.get('experience', '')}"
        )

    # ── Step 3: Get shared Groq client ──
    try:
        client = get_groq_client()
    except ValueError as e:
        return {"error": str(e)}

    # ── Step 4: Single API call for all 4 question types ──
    print("[QuestionGenerator] Generating all questions in one API call...")
    config = MODEL_CONFIGS["question_generator"]
    prompt = get_all_questions_prompt(skills, raw_text, job_description, num_questions)

    raw_text_response = call_with_retry(
        client       = client,
        messages     = [
            {
                "role": "system",
                "content": (
                    "You are an expert interviewer. "
                    "Always respond with ONLY a valid JSON object. "
                    "No explanation, no markdown, no extra text."
                )
            },
            {"role": "user", "content": prompt}
        ],
        model        = config["model"],
        temperature  = config["temperature"],
        max_tokens   = 2048,    # Higher since generating 20 questions at once
        caller_label = "QuestionGenerator:all",
    )

    result = parse_json_response(raw_text_response, "QuestionGenerator:all")

    # ── Step 5: Validate and sanitize response ──
    if result is None or not isinstance(result, dict):
        return {"error": "Failed to generate questions. Please retry."}

    # Ensure all 4 keys exist and are lists
    expected_keys = ["technical", "behavioral", "hr_general", "resume_based"]
    questions = {}
    for key in expected_keys:
        raw = result.get(key, [])
        if isinstance(raw, list) and raw:
            questions[key] = [str(q) for q in raw]
        else:
            questions[key] = [f"Could not generate {key} question {i+1}." for i in range(num_questions)]

    print("[QuestionGenerator] All questions generated successfully!")
    return questions


# ─────────────────────────────────────────────
# SECTION: Save Output to JSON
# ─────────────────────────────────────────────

def save_generated_questions(questions: dict, output_path: str = "output/generated_questions.json") -> None:
    """
    Saves the generated questions dictionary to a JSON file.

    Parameters:
        questions (dict):   The questions dictionary returned by generate_questions().
        output_path (str):  Path to save the JSON file.
                            Defaults to "output/generated_questions.json"

    Returns:
        None
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=4, ensure_ascii=False)
    print(f"[QuestionGenerator] Questions saved to: {output_path}")


# ─────────────────────────────────────────────
# SECTION: Quick Test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    PARSED_RESUME_PATH   = "output/parsed_resume.json"
    JOB_DESCRIPTION_PATH = "data/sample_job_description.txt"
    OUTPUT_PATH          = "output/generated_questions.json"

    if not os.path.exists(PARSED_RESUME_PATH):
        print(f"[QuestionGenerator] ERROR: '{PARSED_RESUME_PATH}' not found. Run resume_parser first.")
        exit(1)

    with open(PARSED_RESUME_PATH, "r", encoding="utf-8") as f:
        parsed_resume = json.load(f)

    if not os.path.exists(JOB_DESCRIPTION_PATH):
        job_description = "We are looking for a Software Engineer with full-stack development experience."
    else:
        with open(JOB_DESCRIPTION_PATH, "r", encoding="utf-8") as f:
            job_description = f.read()

    print("[QuestionGenerator] Starting test generation...")
    result = generate_questions(parsed_resume, job_description)

    if "error" in result:
        print(f"[QuestionGenerator] FAILED: {result['error']}")
    else:
        print("\n[QuestionGenerator] SUCCESS! Generated questions:")
        for q_type, questions in result.items():
            print(f"\n  [{q_type.upper()}]")
            for i, q in enumerate(questions, 1):
                print(f"    {i}. {q}")
        save_generated_questions(result, OUTPUT_PATH)