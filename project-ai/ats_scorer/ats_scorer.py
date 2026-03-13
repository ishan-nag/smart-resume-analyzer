"""
ats_scorer.py — ATS Resume Scoring Module
==========================================
This module scores a candidate's resume against a job description
using a combination of:

    1. Keyword Matching  — compares resume skills vs skills in job description
                           (no API call — runs instantly)
    2. LLM Scoring       — uses ONE Groq API call to evaluate:
                           - Semantic match (overall fit)
                           - Experience match (work history relevance)
                           - Education match (degree and qualification fit)

OPTIMIZATION:
    Previously made 3 separate Groq API calls for semantic, experience,
    and education scoring. Now merged into a single API call that returns
    all three scores in one JSON response — reducing calls from 3 to 1.

Final output includes:
    - An overall ATS score (0-100), weighted average of all 4 categories
    - A per-category breakdown with individual scores and feedback
    - A list of matched and missing keywords
    - A hiring recommendation label

Output format:
    {
        "overall_score": 74,
        "recommendation": "Good Match",
        "breakdown": {
            "keyword_match": {
                "score": 80,
                "matched_keywords": ["python", "docker", "react"],
                "missing_keywords": ["kubernetes", "terraform"],
                "feedback": "Candidate matches 8 out of 10 required skills."
            },
            "semantic_match":   {"score": 75, "feedback": "..."},
            "experience_match": {"score": 70, "feedback": "..."},
            "education_match":  {"score": 90, "feedback": "..."}
        }
    }

For backend integration (Java Spring Boot):
    - Call score_resume(parsed_resume, job_description)
    - Pass the dict from resume_parser directly as parsed_resume
    - Returns a dict → serialize with json.dumps() to send as JSON
    - Optionally call save_ats_result(result, output_path) to persist

Dependencies:
    pip install groq python-dotenv
"""

import os
import json
import re
import glob

# ── Import from shared/ ──
from shared.groq_client import get_groq_client, MODEL_CONFIGS
from shared.retry_handler import call_with_retry, parse_json_response

# ── Import combined prompt ──
try:
    from .prompt_templates import get_combined_llm_scores_prompt
except ImportError:
    from prompt_templates import get_combined_llm_scores_prompt


# ─────────────────────────────────────────────
# SECTION: Scoring Weights
# ─────────────────────────────────────────────

WEIGHTS = {
    "keyword_match":    0.35,
    "semantic_match":   0.35,
    "experience_match": 0.20,
    "education_match":  0.10,
}


# ─────────────────────────────────────────────
# SECTION: Recommendation Labels
# ─────────────────────────────────────────────

def get_recommendation(score: float) -> str:
    """
    Returns a human-readable hiring recommendation label
    based on the overall ATS score.

    Parameters:
        score (float): The overall ATS score (0-100).

    Returns:
        str: One of: "Excellent Match", "Good Match",
                     "Moderate Match", "Weak Match", "Poor Match"
    """
    if score >= 85:
        return "Excellent Match"
    elif score >= 70:
        return "Good Match"
    elif score >= 55:
        return "Moderate Match"
    elif score >= 40:
        return "Weak Match"
    else:
        return "Poor Match"


# ─────────────────────────────────────────────
# SECTION: Keyword Match Scorer
# ─────────────────────────────────────────────

def score_keyword_match(resume_skills: list, job_description: str) -> dict:
    """
    Scores the resume based on keyword/skill overlap with the job description.
    Uses exact and partial string matching — no API call needed.

    Parameters:
        resume_skills (list):   List of skills extracted from resume.
                                Example: ["python", "docker", "react"]
        job_description (str):  Full text of the job description.

    Returns:
        dict: {
            "score":            int (0-100),
            "matched_keywords": list,
            "missing_keywords": list,
            "feedback":         str
        }
    """
    jd_lower = job_description.lower()
    matched  = []
    missing  = []

    for skill in resume_skills:
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, jd_lower):
            matched.append(skill)
        else:
            missing.append(skill)

    # Detect extra JD skills not in resume
    jd_words = re.findall(
        r'\b[a-zA-Z][a-zA-Z0-9+#.]*(?:\s[a-zA-Z][a-zA-Z0-9+#.]*){0,2}\b',
        jd_lower
    )
    jd_skills_detected  = [w for w in set(jd_words) if len(w) > 3]
    resume_skills_lower = [s.lower() for s in resume_skills]
    extra_missing = [
        w for w in jd_skills_detected
        if w not in resume_skills_lower and w not in missing
        and len(w.split()) <= 2
    ][:10]

    all_missing   = list(set(missing + extra_missing))
    total_checked = len(resume_skills) if resume_skills else 1
    score         = max(0, min(100, int((len(matched) / total_checked) * 100)))
    feedback      = (
        f"Candidate matches {len(matched)} out of {total_checked} resume skills "
        f"found in the job description."
    )

    return {
        "score":            score,
        "matched_keywords": matched,
        "missing_keywords": all_missing[:15],
        "feedback":         feedback,
    }


# ─────────────────────────────────────────────
# SECTION: Combined LLM Scorer (1 API call)
# ─────────────────────────────────────────────

def _call_llm_for_all_scores(
    client,
    resume_text: str,
    experience_text: str,
    education_text: str,
    job_description: str
) -> dict:
    """
    Makes a SINGLE Groq API call to get semantic, experience, and
    education match scores all at once.

    Previously this was 3 separate API calls. Now it is 1.

    Parameters:
        client:                 Shared Groq client from shared/groq_client.py
        resume_text (str):      Full raw text of the resume.
        experience_text (str):  Work experience section.
        education_text (str):   Education section.
        job_description (str):  Full job description text.

    Returns:
        dict: {
            "semantic_match":   {"score": int, "feedback": str},
            "experience_match": {"score": int, "feedback": str},
            "education_match":  {"score": int, "feedback": str}
        }
        Returns fallback dict with score=0 for all on failure.
    """
    config = MODEL_CONFIGS["ats_scorer"]
    prompt = get_combined_llm_scores_prompt(
        resume_text, experience_text, education_text, job_description
    )

    raw_text = call_with_retry(
        client       = client,
        messages     = [
            {
                "role": "system",
                "content": (
                    "You are an expert ATS evaluator and recruiter. "
                    "Always respond with ONLY a valid JSON object. "
                    "No explanation, no markdown, no extra text."
                )
            },
            {"role": "user", "content": prompt}
        ],
        model        = config["model"],
        temperature  = config["temperature"],
        max_tokens   = 512,
        caller_label = "ATSScorer:combined",
    )

    data = parse_json_response(raw_text, "ATSScorer:combined")

    # ── Fallback if parsing failed ──
    fallback = {
        "semantic_match":   {"score": 0, "feedback": "Could not evaluate semantic match."},
        "experience_match": {"score": 0, "feedback": "Could not evaluate experience match."},
        "education_match":  {"score": 0, "feedback": "Could not evaluate education match."},
    }

    if data is None or not isinstance(data, dict):
        return fallback

    result = {}
    for key in ["semantic_match", "experience_match", "education_match"]:
        raw = data.get(key, {})
        if isinstance(raw, dict):
            score    = max(0, min(100, int(raw.get("score", 0))))
            feedback = str(raw.get("reason", "No feedback provided."))
            result[key] = {"score": score, "feedback": feedback}
        else:
            result[key] = fallback[key]

    return result


# ─────────────────────────────────────────────
# SECTION: Main Score Function
# ─────────────────────────────────────────────

def score_resume(parsed_resume: dict, job_description: str) -> dict:
    """
    Main function to score a resume against a job description.
    This is the PRIMARY function the backend should call.

    Uses 2 steps:
        Step 1 — Keyword match (no API call, instant)
        Step 2 — ONE Groq API call for semantic + experience + education scores

    Total API calls: 1 (down from 3 previously)

    Parameters:
        parsed_resume (dict):
            The structured resume dictionary returned by resume_parser's
            parse_resume() function. Expected keys used:
                - "skills"     (list) — for keyword matching
                - "raw_text"   (str)  — for semantic scoring
                - "experience" (str)  — for experience match
                - "education"  (str)  — for education match

        job_description (str):
            The full text of the job description.

    Returns:
        dict: A structured ATS scoring result.

        Example return value:
        {
            "overall_score":  74.0,
            "recommendation": "Good Match",
            "breakdown": {
                "keyword_match":    {"score": 80, "matched_keywords": [...],
                                     "missing_keywords": [...], "feedback": "..."},
                "semantic_match":   {"score": 75, "feedback": "..."},
                "experience_match": {"score": 70, "feedback": "..."},
                "education_match":  {"score": 90, "feedback": "..."}
            }
        }

        On failure, returns:
        {
            "error": "Reason for failure"
        }

    Example usage (Python):
        from resume_parser import parse_resume
        from ats_scorer.ats_scorer import score_resume

        resume   = parse_resume("data/sample_resume.pdf")
        job_desc = open("data/sample_job_description.txt").read()
        result   = score_resume(resume, job_desc)
        print(result["overall_score"])
        print(result["recommendation"])
    """

    # ── Step 1: Validate inputs ──
    if not parsed_resume:
        return {"error": "parsed_resume is empty or None."}
    if not job_description or not job_description.strip():
        return {"error": "job_description is empty or None."}

    # ── Step 2: Extract fields ──
    skills     = parsed_resume.get("skills", [])
    raw_text   = parsed_resume.get("raw_text", "")
    experience = parsed_resume.get("experience", "")
    education  = parsed_resume.get("education", "")

    if not raw_text:
        raw_text = f"Skills: {', '.join(skills)}\nExperience: {experience}\nEducation: {education}"
    if not experience.strip():
        experience = raw_text
    if not education.strip():
        education = raw_text

    # ── Step 3: Get shared Groq client ──
    try:
        client = get_groq_client()
    except ValueError as e:
        return {"error": str(e)}

    print("[ATSScorer] Starting ATS scoring...")
    breakdown = {}

    # ── Step 4: Keyword Match (no API call) ──
    print("[ATSScorer] Scoring keyword match...")
    breakdown["keyword_match"] = score_keyword_match(skills, job_description)

    # ── Step 5: Single LLM call for all 3 remaining scores ──
    print("[ATSScorer] Scoring semantic, experience, and education match via single Groq call...")
    llm_scores = _call_llm_for_all_scores(
        client, raw_text, experience, education, job_description
    )
    breakdown["semantic_match"]   = llm_scores["semantic_match"]
    breakdown["experience_match"] = llm_scores["experience_match"]
    breakdown["education_match"]  = llm_scores["education_match"]

    # ── Step 6: Weighted overall score ──
    overall_score = round(sum(
        breakdown[k]["score"] * WEIGHTS[k] for k in WEIGHTS
    ), 1)

    print(f"[ATSScorer] Scoring complete! Overall score: {overall_score}")

    return {
        "overall_score":  overall_score,
        "recommendation": get_recommendation(overall_score),
        "breakdown":      breakdown,
    }


# ─────────────────────────────────────────────
# SECTION: Save Output to JSON
# ─────────────────────────────────────────────

def save_ats_result(result: dict, output_path: str = "output/ats_result.json") -> None:
    """
    Saves the ATS scoring result dictionary to a JSON file.

    Parameters:
        result (dict):      The scoring result returned by score_resume().
        output_path (str):  Path to save the JSON file.
                            Defaults to "output/ats_result.json"

    Returns:
        None
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f"[ATSScorer] ATS result saved to: {output_path}")


# ─────────────────────────────────────────────
# SECTION: Quick Test — scores ALL parsed resumes
# ─────────────────────────────────────────────

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    JOB_DESCRIPTION_PATH = "data/sample_job_description.txt"

    # ── Load job description ──
    if not os.path.exists(JOB_DESCRIPTION_PATH):
        job_description = "We are looking for a Software Engineer with full-stack development experience."
    else:
        with open(JOB_DESCRIPTION_PATH, "r", encoding="utf-8") as f:
            job_description = f.read()

    # ── Auto-detect all parsed resume JSON files ──
    # Matches: output/parsed_resume.json, output/parsed_resume_1.json, etc.
    resume_files = sorted(glob.glob("output/parsed_resume*.json"))

    if not resume_files:
        print("[ATSScorer] ERROR: No parsed resume files found in output/.")
        print("[ATSScorer] Run resume_parser first: python -m resume_parser.resume_parser")
        exit(1)

    print(f"[ATSScorer] Found {len(resume_files)} parsed resume(s): {resume_files}")

    for resume_path in resume_files:

        # ── Derive output path: parsed_resume_1.json → ats_result_1.json ──
        basename    = os.path.basename(resume_path)                   # parsed_resume_1.json
        suffix      = basename.replace("parsed_resume", "ats_result") # ats_result_1.json
        output_path = os.path.join("output", suffix)

        print(f"\n{'='*60}")
        print(f"[ATSScorer] Scoring: {resume_path}")
        print(f"{'='*60}")

        with open(resume_path, "r", encoding="utf-8") as f:
            parsed_resume = json.load(f)

        result = score_resume(parsed_resume, job_description)

        if "error" in result:
            print(f"[ATSScorer] FAILED: {result['error']}")
        else:
            print(f"\n[ATSScorer] SUCCESS!")
            print(f"  Overall Score  : {result['overall_score']} / 100")
            print(f"  Recommendation : {result['recommendation']}")
            print(f"\n  Breakdown:")
            for category, data in result["breakdown"].items():
                print(f"\n    [{category.upper()}]")
                print(f"      Score    : {data['score']} / 100")
                print(f"      Feedback : {data['feedback']}")
                if "matched_keywords" in data:
                    print(f"      Matched  : {data['matched_keywords']}")
                if "missing_keywords" in data:
                    print(f"      Missing  : {data['missing_keywords']}")
            save_ats_result(result, output_path)