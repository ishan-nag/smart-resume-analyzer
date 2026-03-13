"""
ats_scorer/__init__.py
=======================
Exposes the public API of the ats_scorer module.

Backend developer only needs to import from here:

    from ats_scorer import score_resume, save_ats_result

Available functions:
    - score_resume(parsed_resume, job_description) → dict
    - save_ats_result(result, output_path)         → None

Input:
    parsed_resume   (dict) — output of resume_parser's parse_resume()
    job_description (str)  — full job description text

Output:
    {
        "overall_score":  float (0–100),
        "recommendation": str   ("Excellent Match" / "Good Match" /
                                 "Moderate Match" / "Weak Match" / "Poor Match"),
        "breakdown": {
            "keyword_match":    {"score": int, "matched_keywords": list,
                                 "missing_keywords": list, "feedback": str},
            "semantic_match":   {"score": int, "feedback": str},
            "experience_match": {"score": int, "feedback": str},
            "education_match":  {"score": int, "feedback": str}
        }
    }
"""

from ats_scorer.ats_scorer import score_resume, save_ats_result

__all__ = ["score_resume", "save_ats_result"]