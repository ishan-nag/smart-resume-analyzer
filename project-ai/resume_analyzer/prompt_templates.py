"""
prompt_templates.py — Prompt Templates for Resume Analyzer
===========================================================
This module contains all prompt strings used by resume_analyzer.py
to perform LLM-based resume analysis via the Groq API.

Keeping prompts separate from logic makes it easy to:
    - Tweak evaluation criteria without touching analyzer logic
    - Review and improve prompts independently
    - Add new prompt types in the future

This module contains TWO prompts:

    1. get_combined_analysis_prompt()
       — Used by analyze_resume() for ONE LLM call per role.
       — Batches ATS scoring + section feedback + quality score
         into a single prompt to minimize API calls.

    2. get_upgrade_tip_prompt()
       — Used by generate_upgrade_tip() for ONE LLM call total.
       — Takes all role results together and returns a single
         consolidated upgrade tip paragraph for the candidate.
"""


# ─────────────────────────────────────────────
# PROMPT 1: Combined Analysis (1 call per role)
# ─────────────────────────────────────────────

def get_combined_analysis_prompt(
    parsed_resume: dict,
    job_description: str,
    role_title: str,
    required_skills: list,
    nice_to_have_skills: list
) -> str:
    """
    Returns a single combined prompt that asks Groq to evaluate the
    resume against a specific job role across ALL dimensions at once.

    This single prompt replaces what would otherwise be multiple
    separate LLM calls. It batches:
        - ATS scoring   (semantic, experience, education match)
        - Skills gap    (matched vs missing required and nice-to-have skills)
        - Quality score (format, clarity, impact, brevity)
        - Section feedback (experience, education, summary, skills sections)

    Called by: resume_analyzer.analyze_resume()
    API calls: 1 per role

    Parameters:
        parsed_resume (dict):
            The structured resume dict from resume_parser.
            Keys used: raw_text, experience, education, summary, skills

        job_description (str):
            The plain text job description built by
            job_roles.build_job_description() for the target role.

        role_title (str):
            Display name of the role. Example: "Machine Learning Engineer"
            Used inside the prompt so the LLM knows which role to evaluate for.

        required_skills (list):
            List of required skills for the role from job_roles.json.
            Example: ["python", "pytorch", "docker"]

        nice_to_have_skills (list):
            List of nice-to-have skills for the role from job_roles.json.
            Example: ["kubernetes", "mlflow"]

    Returns:
        str: A formatted prompt string ready to send to the Groq API.

    Expected LLM response format (JSON):
        {
            "ats": {
                "overall_score":  78,
                "recommendation": "Good Match",
                "breakdown": {
                    "semantic_match":   {"score": 80, "feedback": "..."},
                    "experience_match": {"score": 70, "feedback": "..."},
                    "education_match":  {"score": 85, "feedback": "..."}
                }
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
                    "feedback":     "Good range of projects but bullet points lack metrics.",
                    "improvements": ["Add quantifiable achievements", "Use action verbs"]
                },
                "education": {
                    "score":        90,
                    "feedback":     "Degree is well-aligned with the role.",
                    "improvements": []
                },
                "summary": {
                    "score":        60,
                    "feedback":     "Summary is generic and does not highlight key strengths.",
                    "improvements": ["Tailor summary to ML roles", "Mention top tools"]
                },
                "skills": {
                    "score":        80,
                    "feedback":     "Strong core skills listed.",
                    "improvements": ["Add MLflow and Kubernetes to skills section"]
                }
            }
        }

    Note:
        skills_gap is NOT part of the LLM prompt — it is computed
        separately in resume_analyzer.py using regex matching
        (zero API calls). Only ats, quality_score, and section_feedback
        come from this LLM call.
    """
    # Extract fields from parsed_resume for the prompt
    raw_text   = parsed_resume.get("raw_text", "")
    experience = parsed_resume.get("experience", "")
    education  = parsed_resume.get("education", "")
    summary    = parsed_resume.get("summary", "")
    skills     = parsed_resume.get("skills", [])

    # Fallbacks if sections are empty
    if not experience.strip():
        experience = raw_text
    if not education.strip():
        education = raw_text

    return f"""You are an expert resume evaluator and career coach.

Evaluate the candidate's resume for the role of "{role_title}".

---
FULL RESUME:
{raw_text}

WORK EXPERIENCE SECTION:
{experience}

EDUCATION SECTION:
{education}

SUMMARY SECTION:
{summary}

CANDIDATE SKILLS:
{", ".join(skills)}

JOB DESCRIPTION:
{job_description}

REQUIRED SKILLS FOR THIS ROLE:
{", ".join(required_skills)}

NICE TO HAVE SKILLS FOR THIS ROLE:
{", ".join(nice_to_have_skills)}
---

Evaluate the resume across the following three sections.
Respond ONLY with a valid JSON object. No explanation, no markdown, no extra text.

SECTION 1 — ATS SCORING:
Score how well this resume matches the job description.
Evaluate semantic match (overall fit), experience match (work history relevance),
and education match (degree and qualification fit). Do NOT include keyword_match here.

SECTION 2 — QUALITY SCORE:
Score the overall resume quality regardless of the role.
Evaluate: format (structure and layout), clarity (readability),
impact (strength of achievements), brevity (conciseness).
Each sub-score is 0-100. Overall is the average of the four.

SECTION 3 — SECTION FEEDBACK:
Give specific feedback for each of these four resume sections:
experience, education, summary, skills.
For each section provide: a score (0-100), one or two sentence feedback,
and a list of 0-3 concrete improvement suggestions.
If a section is strong, improvements can be an empty list.

The JSON must have exactly this structure:
{{
    "ats": {{
        "overall_score":  <0-100 float>,
        "recommendation": "<Excellent Match|Good Match|Moderate Match|Weak Match|Poor Match>",
        "breakdown": {{
            "semantic_match":   {{"score": <0-100>, "feedback": "<one or two sentences>"}},
            "experience_match": {{"score": <0-100>, "feedback": "<one or two sentences>"}},
            "education_match":  {{"score": <0-100>, "feedback": "<one or two sentences>"}}
        }}
    }},
    "quality_score": {{
        "overall": <0-100 int>,
        "breakdown": {{
            "format":  <0-100>,
            "clarity": <0-100>,
            "impact":  <0-100>,
            "brevity": <0-100>
        }}
    }},
    "section_feedback": {{
        "experience": {{
            "score":        <0-100>,
            "feedback":     "<one or two sentences>",
            "improvements": ["<suggestion 1>", "<suggestion 2>"]
        }},
        "education": {{
            "score":        <0-100>,
            "feedback":     "<one or two sentences>",
            "improvements": ["<suggestion 1>"]
        }},
        "summary": {{
            "score":        <0-100>,
            "feedback":     "<one or two sentences>",
            "improvements": ["<suggestion 1>", "<suggestion 2>"]
        }},
        "skills": {{
            "score":        <0-100>,
            "feedback":     "<one or two sentences>",
            "improvements": ["<suggestion 1>"]
        }}
    }}
}}
"""


# ─────────────────────────────────────────────
# PROMPT 2: Upgrade Tip (1 call total, after all roles)
# ─────────────────────────────────────────────

def get_upgrade_tip_prompt(
    parsed_resume: dict,
    all_role_results: list
) -> str:
    """
    Returns a prompt that asks Groq to generate a single consolidated
    resume upgrade tip paragraph based on results across all analyzed roles.

    Called ONCE by resume_analyzer.generate_upgrade_tip() after all
    analyze_resume() calls are complete. This is the final LLM call
    in the entire pipeline.

    Called by: resume_analyzer.generate_upgrade_tip()
    API calls: 1 total (called once, not per role)

    Parameters:
        parsed_resume (dict):
            The structured resume dict from resume_parser.
            Keys used: name, skills, summary

        all_role_results (list):
            List of dicts — one per role — as returned by analyze_resume().
            Each dict contains: role, ats, skills_gap, quality_score,
            section_feedback.
            Used to identify cross-role patterns in weaknesses.

    Returns:
        str: A formatted prompt string ready to send to the Groq API.

    Expected LLM response format (JSON):
        {
            "upgrade_tip": "Your resume shows strong Python fundamentals
                            across all roles but consistently lacks cloud
                            and deployment skills like Docker and Kubernetes.
                            Adding these to your experience section with
                            concrete project examples would significantly
                            improve your ATS scores across Data & AI and
                            Infrastructure roles."
        }
    """
    candidate_name   = parsed_resume.get("name", "The candidate")
    candidate_skills = parsed_resume.get("skills", [])
    candidate_summary = parsed_resume.get("summary", "")

    # Build a compact summary of each role's results for the prompt
    role_summaries = []
    for result in all_role_results:
        # Skip failed role results
        if "error" in result:
            continue

        role_title   = result.get("role", {}).get("title", "Unknown Role")
        ats_score    = result.get("ats", {}).get("overall_score", 0)
        missing      = result.get("skills_gap", {}).get("missing", [])
        nth_missing  = result.get("skills_gap", {}).get("nice_to_have_missing", [])
        quality      = result.get("quality_score", {}).get("overall", 0)

        role_summaries.append(
            f"- {role_title}: ATS Score={ats_score}/100, Quality={quality}/100, "
            f"Missing Skills={missing}, Nice-to-Have Missing={ nth_missing}"
        )

    role_summary_text = "\n".join(role_summaries) if role_summaries else "No role results available."

    return f"""You are an expert career coach and resume consultant.

A candidate has analyzed their resume against multiple job roles.
Based on the results below, write a single concise upgrade tip paragraph
that will help them improve their resume the most across all roles.

CANDIDATE NAME: {candidate_name}
CANDIDATE SKILLS: {", ".join(candidate_skills)}
CANDIDATE SUMMARY: {candidate_summary}

ROLE ANALYSIS RESULTS:
{role_summary_text}

Write ONE paragraph (3-5 sentences) that:
    - Identifies the most common or impactful gap across all roles
    - Gives specific, actionable advice the candidate can act on immediately
    - Mentions specific skills, sections, or resume improvements by name
    - Is encouraging and professional in tone
    - Does NOT repeat generic advice like "tailor your resume"

Respond ONLY with a valid JSON object. No explanation, no markdown, no extra text.
The JSON must have exactly this structure:
{{
    "upgrade_tip": "<your 3-5 sentence upgrade tip paragraph here>"
}}
"""