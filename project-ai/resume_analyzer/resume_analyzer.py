"""
resume_analyzer.py — Core Resume Analysis Module
=================================================
This is the PRIMARY module the backend calls to analyze a resume
against a specific job role.

This module is responsible for:
    - Orchestrating the full analysis pipeline for one role at a time
    - Batching ATS scoring + section feedback + quality score into 1 LLM call
    - Computing skills gap using regex (zero API calls)
    - Generating a cross-role upgrade tip in 1 final LLM call

PUBLIC FUNCTIONS (backend calls these):
    1. analyze_resume(parsed_resume, role_id)
       — Call once per role. Returns full analysis for that role.
       — 1 LLM call per role.

    2. generate_upgrade_tip(all_role_results, parsed_resume)
       — Call ONCE after all analyze_resume() calls are done.
       — 1 LLM call total.

API call budget per full session (4 roles):
    - 1 call  : parse_resume()            (resume_parser)
    - 4 calls : analyze_resume() × 4      (this module)
    - 1 call  : generate_upgrade_tip()    (this module)
    Total     : 6 calls

For backend integration (Java Spring Boot):
    Step 1 — parse_resume(pdf_path)                          → parsed_resume
    Step 2 — for each role_id: analyze_resume(parsed_resume, role_id)
                                                             → list of role results
    Step 3 — generate_upgrade_tip(all_role_results, parsed_resume)
                                                             → upgrade tip
    Step 4 — best_match = max(all_role_results,
                 key=lambda r: r["ats"]["overall_score"])
    Step 5 — return full JSON to frontend

Dependencies:
    pip install groq python-dotenv
"""

import os
import re
import json
import glob

# ── Shared utilities ──
from shared.groq_client import get_groq_client, MODEL_CONFIGS
from shared.retry_handler import call_with_retry, parse_json_response

# ── Job roles helpers ──
from job_roles.job_roles import get_role_by_id, build_job_description

# ── Prompts ──
try:
    from .prompt_templates import get_combined_analysis_prompt, get_upgrade_tip_prompt
except ImportError:
    from prompt_templates import get_combined_analysis_prompt, get_upgrade_tip_prompt


# ─────────────────────────────────────────────
# SECTION: Skills Gap (regex, zero API calls)
# ─────────────────────────────────────────────

def _compute_skills_gap(
    resume_skills: list,
    required_skills: list,
    nice_to_have_skills: list
) -> dict:
    """
    Computes the skills gap between resume skills and role skills.
    Uses simple string matching — no API call needed.

    Parameters:
        resume_skills (list):       Skills extracted from the resume.
                                    Example: ["python", "docker", "react"]
        required_skills (list):     Required skills from job_roles.json.
                                    Example: ["python", "pytorch", "docker"]
        nice_to_have_skills (list): Nice-to-have skills from job_roles.json.
                                    Example: ["kubernetes", "mlflow"]

    Returns:
        dict: {
            "matched":              list[str] — required skills found in resume,
            "missing":              list[str] — required skills NOT in resume,
            "nice_to_have_missing": list[str] — nice-to-have skills NOT in resume
        }
    """
    resume_lower = [s.lower().strip() for s in resume_skills]

    matched = []
    missing = []

    for skill in required_skills:
        pattern = r'\b' + re.escape(skill.lower().strip()) + r'\b'
        if any(re.search(pattern, rs) for rs in resume_lower):
            matched.append(skill)
        else:
            missing.append(skill)

    nth_missing = []
    for skill in nice_to_have_skills:
        pattern = r'\b' + re.escape(skill.lower().strip()) + r'\b'
        if not any(re.search(pattern, rs) for rs in resume_lower):
            nth_missing.append(skill)

    return {
        "matched":              matched,
        "missing":              missing,
        "nice_to_have_missing": nth_missing,
    }


# ─────────────────────────────────────────────
# SECTION: LLM Call — Combined Analysis
# ─────────────────────────────────────────────

def _call_llm_for_analysis(
    client,
    parsed_resume: dict,
    job_description: str,
    role_title: str,
    required_skills: list,
    nice_to_have_skills: list
) -> dict:
    """
    Makes ONE Groq API call to get ATS scores, quality score,
    and section feedback all at once for a single role.

    Parameters:
        client:                     Shared Groq client.
        parsed_resume (dict):       Structured resume dict from resume_parser.
        job_description (str):      Plain text JD built from job_roles.json.
        role_title (str):           Display name of the role.
        required_skills (list):     Required skills for the role.
        nice_to_have_skills (list): Nice-to-have skills for the role.

    Returns:
        dict: Raw parsed JSON from the LLM containing:
              ats, quality_score, section_feedback.
              Returns a fallback error dict if the LLM call fails.
    """
    config = MODEL_CONFIGS.get("resume_analyzer", MODEL_CONFIGS.get("default", {}))
    prompt = get_combined_analysis_prompt(
        parsed_resume, job_description, role_title,
        required_skills, nice_to_have_skills
    )

    raw_text = call_with_retry(
        client       = client,
        messages     = [
            {
                "role": "system",
                "content": (
                    "You are an expert resume evaluator and career coach. "
                    "Always respond with ONLY a valid JSON object. "
                    "No explanation, no markdown, no extra text."
                )
            },
            {"role": "user", "content": prompt}
        ],
        model        = config.get("model", "llama-3.3-70b-versatile"),
        temperature  = config.get("temperature", 0.3),
        max_tokens   = 1500,
        caller_label = f"ResumeAnalyzer:analyze:{role_title}",
    )

    data = parse_json_response(raw_text, f"ResumeAnalyzer:analyze:{role_title}")

    if data is None or not isinstance(data, dict):
        return {"error": f"LLM returned invalid response for role: {role_title}"}

    return data


# ─────────────────────────────────────────────
# SECTION: Response Validator + Fallback Builder
# ─────────────────────────────────────────────

def _build_safe_analysis(llm_data: dict) -> dict:
    """
    Validates the LLM response and fills in safe fallback values
    for any missing or malformed fields.

    Ensures the final analyze_resume() output always has the
    correct structure even if the LLM returns partial data.

    Parameters:
        llm_data (dict): Raw parsed JSON from _call_llm_for_analysis().

    Returns:
        dict: Validated and sanitized analysis dict containing:
              ats, quality_score, section_feedback — all with correct types.
    """

    # ── ATS fallback ──
    default_ats = {
        "overall_score":  0.0,
        "recommendation": "Poor Match",
        "breakdown": {
            "semantic_match":   {"score": 0, "feedback": "Could not evaluate."},
            "experience_match": {"score": 0, "feedback": "Could not evaluate."},
            "education_match":  {"score": 0, "feedback": "Could not evaluate."},
        }
    }

    # ── Quality score fallback ──
    default_quality = {
        "overall": 0,
        "breakdown": {"format": 0, "clarity": 0, "impact": 0, "brevity": 0}
    }

    # ── Section feedback fallback ──
    default_section = {
        "score": 0,
        "feedback": "Could not evaluate this section.",
        "improvements": []
    }
    default_feedback = {
        "experience": dict(default_section),
        "education":  dict(default_section),
        "summary":    dict(default_section),
        "skills":     dict(default_section),
    }

    # ── Extract and validate ATS ──
    raw_ats = llm_data.get("ats", {})
    if not isinstance(raw_ats, dict):
        raw_ats = {}

    ats_score = raw_ats.get("overall_score", 0)
    try:
        ats_score = round(float(ats_score), 1)
        ats_score = max(0.0, min(100.0, ats_score))
    except (ValueError, TypeError):
        ats_score = 0.0

    ats_recommendation = str(raw_ats.get("recommendation", "Poor Match"))

    raw_breakdown = raw_ats.get("breakdown", {})
    if not isinstance(raw_breakdown, dict):
        raw_breakdown = {}

    ats_breakdown = {}
    for key in ["semantic_match", "experience_match", "education_match"]:
        raw = raw_breakdown.get(key, {})
        if isinstance(raw, dict):
            score = max(0, min(100, int(raw.get("score", 0))))
            feedback = str(raw.get("feedback", "No feedback provided."))
            ats_breakdown[key] = {"score": score, "feedback": feedback}
        else:
            ats_breakdown[key] = default_ats["breakdown"][key]

    ats = {
        "overall_score":  ats_score,
        "recommendation": ats_recommendation,
        "breakdown":      ats_breakdown,
    }

    # ── Extract and validate quality score ──
    raw_quality = llm_data.get("quality_score", {})
    if not isinstance(raw_quality, dict):
        raw_quality = {}

    quality_overall = raw_quality.get("overall", 0)
    try:
        quality_overall = max(0, min(100, int(quality_overall)))
    except (ValueError, TypeError):
        quality_overall = 0

    raw_qb = raw_quality.get("breakdown", {})
    if not isinstance(raw_qb, dict):
        raw_qb = {}

    quality_breakdown = {}
    for key in ["format", "clarity", "impact", "brevity"]:
        try:
            quality_breakdown[key] = max(0, min(100, int(raw_qb.get(key, 0))))
        except (ValueError, TypeError):
            quality_breakdown[key] = 0

    quality_score = {
        "overall":   quality_overall,
        "breakdown": quality_breakdown,
    }

    # ── Extract and validate section feedback ──
    raw_feedback = llm_data.get("section_feedback", {})
    if not isinstance(raw_feedback, dict):
        raw_feedback = {}

    section_feedback = {}
    for section in ["experience", "education", "summary", "skills"]:
        raw = raw_feedback.get(section, {})
        if isinstance(raw, dict):
            score = raw.get("score", 0)
            try:
                score = max(0, min(100, int(score)))
            except (ValueError, TypeError):
                score = 0
            feedback     = str(raw.get("feedback", "No feedback provided."))
            improvements = raw.get("improvements", [])
            if not isinstance(improvements, list):
                improvements = []
            improvements = [str(i) for i in improvements[:3]]
            section_feedback[section] = {
                "score":        score,
                "feedback":     feedback,
                "improvements": improvements,
            }
        else:
            section_feedback[section] = dict(default_feedback[section])

    return {
        "ats":              ats,
        "quality_score":    quality_score,
        "section_feedback": section_feedback,
    }


# ─────────────────────────────────────────────
# SECTION: analyze_resume() — PRIMARY FUNCTION
# ─────────────────────────────────────────────

def analyze_resume(parsed_resume: dict, role_id: str) -> dict:
    """
    Analyzes a parsed resume against a specific job role.
    This is the PRIMARY function the backend calls — once per role.

    Makes exactly 1 Groq API call per role, batching:
        - ATS scoring (semantic, experience, education match)
        - Quality score (format, clarity, impact, brevity)
        - Section feedback (experience, education, summary, skills)

    Skills gap is computed with regex — no API call needed.

    Parameters:
        parsed_resume (dict):
            The structured resume dict returned by resume_parser.parse_resume().
            Expected keys:
                - "name"       (str)  — candidate name
                - "skills"     (list) — list of skill strings
                - "raw_text"   (str)  — full resume text
                - "experience" (str)  — work experience section
                - "education"  (str)  — education section
                - "summary"    (str)  — professional summary section

        role_id (str):
            The unique role identifier from job_roles.json.
            Example: "ml_engineer", "frontend_engineer", "devops_engineer"
            See get_all_roles() for the full list of valid role IDs.

    Returns:
        dict: Full analysis result for this role.

        Structure:
        {
            "role": {
                "id":       "ml_engineer",
                "title":    "Machine Learning Engineer",
                "category": "Data & AI"
            },
            "ats": {
                "overall_score":  78.0,
                "recommendation": "Good Match",
                "breakdown": {
                    "semantic_match":   {"score": 80, "feedback": "..."},
                    "experience_match": {"score": 70, "feedback": "..."},
                    "education_match":  {"score": 85, "feedback": "..."}
                }
            },
            "skills_gap": {
                "matched":              ["python", "pytorch", "docker"],
                "missing":              ["tensorflow", "mlflow"],
                "nice_to_have_missing": ["kubernetes", "airflow"]
            },
            "quality_score": {
                "overall": 72,
                "breakdown": {
                    "format":  75,
                    "clarity": 70,
                    "impact":  68,
                    "brevity": 80
                }
            },
            "section_feedback": {
                "experience": {
                    "score":        75,
                    "feedback":     "Good range but lacks metrics.",
                    "improvements": ["Add quantifiable achievements"]
                },
                "education":  {"score": 90, "feedback": "...", "improvements": []},
                "summary":    {"score": 60, "feedback": "...", "improvements": [...]},
                "skills":     {"score": 80, "feedback": "...", "improvements": [...]}
            }
        }

        On failure, returns:
        {
            "error": "Reason for failure"
        }

    Example usage:
        from resume_parser.resume_parser import parse_resume
        from resume_analyzer.resume_analyzer import analyze_resume

        parsed = parse_resume("data/sample_resume.pdf")
        result = analyze_resume(parsed, role_id="ml_engineer")

        if "error" not in result:
            print(result["ats"]["overall_score"])
            print(result["skills_gap"]["missing"])
    """

    # ── Step 1: Validate inputs ──
    if not parsed_resume:
        return {"error": "parsed_resume is empty or None."}
    if not role_id or not role_id.strip():
        return {"error": "role_id is empty or None."}

    # ── Step 2: Load role from job_roles.json ──
    role = get_role_by_id(role_id)
    if "error" in role:
        return {"error": f"Invalid role_id: '{role_id}'. {role['error']}"}

    role_title          = role["title"]
    role_category       = role["category"]
    required_skills     = role.get("required_skills", [])
    nice_to_have_skills = role.get("nice_to_have_skills", [])

    print(f"[ResumeAnalyzer] Analyzing resume for role: '{role_title}'...")

    # ── Step 3: Build job description string ──
    job_description = build_job_description(role)

    # ── Step 4: Compute skills gap (regex, no API call) ──
    resume_skills = parsed_resume.get("skills", [])
    skills_gap    = _compute_skills_gap(
        resume_skills, required_skills, nice_to_have_skills
    )
    print(f"[ResumeAnalyzer] Skills gap computed. "
          f"Matched: {len(skills_gap['matched'])}, "
          f"Missing: {len(skills_gap['missing'])}")

    # ── Step 5: Get shared Groq client ──
    try:
        client = get_groq_client()
    except ValueError as e:
        return {"error": str(e)}

    # ── Step 6: Single LLM call — ATS + quality + section feedback ──
    print(f"[ResumeAnalyzer] Calling Groq API for combined analysis...")
    llm_data = _call_llm_for_analysis(
        client, parsed_resume, job_description,
        role_title, required_skills, nice_to_have_skills
    )

    if "error" in llm_data:
        return {"error": llm_data["error"]}

    # ── Step 7: Validate and sanitize LLM response ──
    safe_data = _build_safe_analysis(llm_data)

    print(f"[ResumeAnalyzer] Analysis complete for '{role_title}'. "
          f"ATS Score: {safe_data['ats']['overall_score']}")

    # ── Step 8: Assemble final result ──
    return {
        "role": {
            "id":       role_id,
            "title":    role_title,
            "category": role_category,
        },
        "ats":              safe_data["ats"],
        "skills_gap":       skills_gap,
        "quality_score":    safe_data["quality_score"],
        "section_feedback": safe_data["section_feedback"],
    }


# ─────────────────────────────────────────────
# SECTION: generate_upgrade_tip() — FINAL CALL
# ─────────────────────────────────────────────

def generate_upgrade_tip(all_role_results: list, parsed_resume: dict) -> dict:
    """
    Generates a single consolidated resume upgrade tip based on results
    across all analyzed roles. Called ONCE after all analyze_resume() calls.

    Makes exactly 1 Groq API call total.

    Parameters:
        all_role_results (list):
            List of dicts returned by analyze_resume(), one per role.
            Example: [result_ml_engineer, result_devops, result_backend]
            Failed results (dicts with "error" key) are automatically skipped.

        parsed_resume (dict):
            The structured resume dict from resume_parser.parse_resume().
            Keys used: name, skills, summary.

    Returns:
        dict: {
            "upgrade_tip": "Your resume shows strong Python fundamentals
                            across all roles but consistently lacks cloud
                            and deployment skills..."
        }

        On failure, returns:
        {
            "upgrade_tip": "Focus on adding missing skills to your resume
                            and quantifying your achievements."
        }
        Note: upgrade_tip always returns a dict with the "upgrade_tip" key,
        even on failure — so the frontend can always render something.

    Example usage:
        from resume_analyzer.resume_analyzer import analyze_resume, generate_upgrade_tip

        results = []
        for role_id in ["ml_engineer", "data_scientist", "data_engineer"]:
            results.append(analyze_resume(parsed_resume, role_id))

        tip = generate_upgrade_tip(results, parsed_resume)
        print(tip["upgrade_tip"])
    """

    # ── Step 1: Validate inputs ──
    if not all_role_results:
        return {"upgrade_tip": "No role results available to generate an upgrade tip."}

    if not parsed_resume:
        return {"upgrade_tip": "Resume data unavailable. Please re-upload your resume."}

    # ── Step 2: Filter out failed results ──
    valid_results = [r for r in all_role_results if "error" not in r]
    if not valid_results:
        return {"upgrade_tip": "All role analyses failed. Please try again."}

    print(f"[ResumeAnalyzer] Generating upgrade tip from {len(valid_results)} role result(s)...")

    # ── Step 3: Get shared Groq client ──
    try:
        client = get_groq_client()
    except ValueError as e:
        return {"upgrade_tip": "Could not connect to AI service. Please try again."}

    # ── Step 4: Build prompt and call LLM ──
    config = MODEL_CONFIGS.get("resume_analyzer", MODEL_CONFIGS.get("default", {}))
    prompt = get_upgrade_tip_prompt(parsed_resume, valid_results)

    raw_text = call_with_retry(
        client       = client,
        messages     = [
            {
                "role": "system",
                "content": (
                    "You are an expert career coach and resume consultant. "
                    "Always respond with ONLY a valid JSON object. "
                    "No explanation, no markdown, no extra text."
                )
            },
            {"role": "user", "content": prompt}
        ],
        model        = config.get("model", "llama-3.3-70b-versatile"),
        temperature  = config.get("temperature", 0.4),
        max_tokens   = 512,
        caller_label = "ResumeAnalyzer:upgrade_tip",
    )

    # ── Step 5: Parse and validate response ──
    data = parse_json_response(raw_text, "ResumeAnalyzer:upgrade_tip")

    if data is None or not isinstance(data, dict):
        return {
            "upgrade_tip": (
                "Focus on adding missing skills to your resume "
                "and quantifying your achievements with measurable results."
            )
        }

    tip = data.get("upgrade_tip", "")
    if not tip or not isinstance(tip, str) or not tip.strip():
        return {
            "upgrade_tip": (
                "Focus on adding missing skills to your resume "
                "and quantifying your achievements with measurable results."
            )
        }

    print("[ResumeAnalyzer] Upgrade tip generated successfully.")
    return {"upgrade_tip": tip.strip()}


# ─────────────────────────────────────────────
# SECTION: Save Output to JSON
# ─────────────────────────────────────────────

def save_analysis_result(result: dict, output_path: str) -> None:
    """
    Saves an analyze_resume() result dict to a JSON file.

    Parameters:
        result (dict):      The result returned by analyze_resume().
        output_path (str):  Path to save the JSON file.
                            Example: "output/analysis_ml_engineer.json"

    Returns:
        None
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f"[ResumeAnalyzer] Result saved to: {output_path}")


# ─────────────────────────────────────────────
# SECTION: Quick Test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # ── Test roles ──
    TEST_ROLE_IDS = ["ml_engineer", "backend_engineer"]

    # ── Auto-detect parsed resume files ──
    resume_files = sorted(glob.glob("output/parsed_resume*.json"))

    if not resume_files:
        print("[ResumeAnalyzer] ERROR: No parsed resume files found in output/.")
        print("[ResumeAnalyzer] Run resume_parser first: python -m resume_parser.resume_parser")
        exit(1)

    # ── Use the first parsed resume for testing ──
    resume_path = resume_files[0]
    print(f"[ResumeAnalyzer] Using resume: {resume_path}")

    with open(resume_path, "r", encoding="utf-8") as f:
        parsed_resume = json.load(f)

    all_results = []

    for role_id in TEST_ROLE_IDS:
        print(f"\n{'='*60}")
        print(f"[ResumeAnalyzer] Testing role: {role_id}")
        print(f"{'='*60}")

        result = analyze_resume(parsed_resume, role_id)

        if "error" in result:
            print(f"[ResumeAnalyzer] FAILED: {result['error']}")
        else:
            all_results.append(result)
            print(f"\n  Role      : {result['role']['title']}")
            print(f"  ATS Score : {result['ats']['overall_score']} / 100")
            print(f"  Match     : {result['ats']['recommendation']}")
            print(f"  Matched Skills   : {result['skills_gap']['matched']}")
            print(f"  Missing Skills   : {result['skills_gap']['missing']}")
            print(f"  Quality Score    : {result['quality_score']['overall']} / 100")
            print(f"\n  Section Feedback:")
            for section, data in result["section_feedback"].items():
                print(f"    [{section.upper()}] Score: {data['score']} — {data['feedback']}")
                if data["improvements"]:
                    for tip in data["improvements"]:
                        print(f"      → {tip}")

            save_analysis_result(result, f"output/analysis_{role_id}.json")

    # ── Test generate_upgrade_tip ──
    if all_results:
        print(f"\n{'='*60}")
        print("[ResumeAnalyzer] Testing generate_upgrade_tip()...")
        print(f"{'='*60}")
        tip = generate_upgrade_tip(all_results, parsed_resume)
        print(f"\n  Upgrade Tip:\n  {tip['upgrade_tip']}")
        save_analysis_result(tip, "output/upgrade_tip.json")