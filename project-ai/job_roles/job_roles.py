"""
job_roles.py — Job Roles Data Module
=====================================
This module loads and serves job role data from data/job_roles.json.

Responsibilities:
    - Load the 28 job roles from the JSON file once at import time
    - Provide clean functions to query roles by ID or get a full list
    - Convert role dicts into plain job description strings for LLM prompts

IMPORTANT:
    - Zero API calls — this module only reads a local JSON file
    - All functions return plain Python dicts or strings (JSON-serializable)
    - The backend should call get_all_roles() to populate frontend dropdowns
    - The backend should call get_role_by_id() before calling analyze_resume()

For backend integration (Java Spring Boot):
    - Call get_all_roles()           → send to frontend for dropdown
    - Call get_role_by_id(role_id)   → verify a role exists before analysis
    - build_job_description() is used internally by ats_scorer and resume_analyzer
      — the backend does NOT need to call this directly

Dependencies:
    None — standard library only (json, os)
"""

import os
import json


# ─────────────────────────────────────────────
# SECTION: Load JSON Data
# ─────────────────────────────────────────────

def _load_job_roles() -> list:
    """
    Loads all job roles from data/job_roles.json.
    Called once at module level — not called again after that.

    Returns:
        list: A list of role dicts loaded from the JSON file.
              Returns an empty list if the file is missing or malformed.
    """
    # Build path relative to project-ai/ root
    base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    roles_path = os.path.join(base_dir, "data", "job_roles.json")

    if not os.path.exists(roles_path):
        print(f"[JobRoles] ERROR: job_roles.json not found at {roles_path}")
        return []

    with open(roles_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    roles = data.get("roles", [])
    print(f"[JobRoles] Loaded {len(roles)} job roles from job_roles.json")
    return roles


# Load once when the module is first imported
_ALL_ROLES: list = _load_job_roles()

# Build a lookup dict keyed by role id for O(1) access
_ROLES_BY_ID: dict = {role["id"]: role for role in _ALL_ROLES}


# ─────────────────────────────────────────────
# SECTION: Public Functions
# ─────────────────────────────────────────────

def get_all_roles() -> list:
    """
    Returns a lightweight list of all 28 job roles.
    Designed for the frontend dropdown — does NOT include
    required_skills or nice_to_have_skills to keep payload small.

    Parameters:
        None

    Returns:
        list of dicts, each containing:
            - id               (str)  — unique role identifier, e.g. "ml_engineer"
            - title            (str)  — display name, e.g. "Machine Learning Engineer"
            - category         (str)  — group name, e.g. "Data & AI"
            - experience_level (str)  — e.g. "Mid-level", "Senior", "Entry-level"

    Example return value:
        [
            {
                "id":               "ml_engineer",
                "title":            "Machine Learning Engineer",
                "category":         "Data & AI",
                "experience_level": "Mid-level"
            },
            ...
        ]

    Example usage:
        from job_roles.job_roles import get_all_roles
        roles = get_all_roles()
        # Pass roles list to frontend for the role selection dropdown
    """
    return [
        {
            "id":               role["id"],
            "title":            role["title"],
            "category":         role["category"],
            "experience_level": role["experience_level"],
        }
        for role in _ALL_ROLES
    ]


def get_role_by_id(role_id: str) -> dict:
    """
    Returns the full role dict for a given role ID.
    Includes required_skills, nice_to_have_skills, and description.

    Parameters:
        role_id (str): The unique role identifier string.
                       Example: "ml_engineer", "frontend_engineer"

    Returns:
        dict: Full role dictionary with all fields:
            - id, title, category, experience_level
            - description          (str)
            - required_skills      (list of str)
            - nice_to_have_skills  (list of str)

        If the role_id is not found, returns:
            {"error": "Role not found: <role_id>"}

    Example usage:
        from job_roles.job_roles import get_role_by_id
        role = get_role_by_id("ml_engineer")
        if "error" not in role:
            print(role["required_skills"])
    """
    role = _ROLES_BY_ID.get(role_id)

    if role is None:
        return {"error": f"Role not found: {role_id}"}

    return role


def build_job_description(role: dict) -> str:
    """
    Converts a role dict into a plain text job description string.
    Used internally by ats_scorer and resume_analyzer as LLM prompt input.
    The backend does NOT need to call this directly.

    Parameters:
        role (dict): A full role dict as returned by get_role_by_id().
                     Must contain: title, description,
                                   required_skills, nice_to_have_skills.

    Returns:
        str: A formatted job description string ready for LLM prompts.

        Returns an empty string if the role dict is empty or has an error key.

    Example return value:
        \"\"\"
        Job Title: Machine Learning Engineer

        About the Role:
        Build, train, and deploy machine learning models at scale...

        Required Skills:
        python, pytorch, tensorflow, scikit-learn, docker...

        Nice to Have:
        kubernetes, mlflow, airflow...
        \"\"\"

    Example usage:
        from job_roles.job_roles import get_role_by_id, build_job_description
        role = get_role_by_id("ml_engineer")
        jd   = build_job_description(role)
        # Pass jd string to score_resume() or analyze_resume()
    """
    if not role or "error" in role:
        return ""

    title            = role.get("title", "")
    description      = role.get("description", "")
    required_skills  = role.get("required_skills", [])
    nice_to_have     = role.get("nice_to_have_skills", [])
    experience_level = role.get("experience_level", "")

    jd = f"""Job Title: {title}
Experience Level: {experience_level}

About the Role:
{description}

Required Skills:
{", ".join(required_skills)}

Nice to Have:
{", ".join(nice_to_have)}
"""
    return jd


# ─────────────────────────────────────────────
# SECTION: Quick Test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("\n" + "="*60)
    print("TEST 1: get_all_roles()")
    print("="*60)
    all_roles = get_all_roles()
    print(f"Total roles loaded: {len(all_roles)}")
    for r in all_roles:
        print(f"  [{r['category']}] {r['id']} — {r['title']} ({r['experience_level']})")

    print("\n" + "="*60)
    print("TEST 2: get_role_by_id('ml_engineer')")
    print("="*60)
    role = get_role_by_id("ml_engineer")
    print(f"  Title            : {role['title']}")
    print(f"  Category         : {role['category']}")
    print(f"  Experience Level : {role['experience_level']}")
    print(f"  Required Skills  : {role['required_skills']}")
    print(f"  Nice to Have     : {role['nice_to_have_skills']}")

    print("\n" + "="*60)
    print("TEST 3: get_role_by_id('invalid_role')")
    print("="*60)
    bad_role = get_role_by_id("invalid_role")
    print(f"  Result: {bad_role}")

    print("\n" + "="*60)
    print("TEST 4: build_job_description(ml_engineer)")
    print("="*60)
    jd = build_job_description(role)
    print(jd)