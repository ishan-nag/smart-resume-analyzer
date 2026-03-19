# Smart Resume Analyzer — AI Module

The AI module handles resume parsing and resume vs. job role analysis. It is written in Python and uses the Groq API (free tier) for LLM calls. The backend calls functions from this module — the frontend never touches it directly.

---

## Team

| Role | Technology |
|---|---|
| AI/ML | Python — this module |
| Backend | Java Spring Boot |
| Frontend | Separate teammate |

---

## Project Repository Structure

```
smart-resume-analyzer/
├── project-ai/         ← Python AI module (AI/ML teammate)
├── project-backend/    ← Java Spring Boot (backend teammate)
└── project-frontend/   ← Frontend (frontend teammate)
```

---

## For Teammates — Getting Started with the Repo

### Step 1 — Clone the repository to your Desktop

Open a terminal (Command Prompt, PowerShell, or Git Bash) and run:

```bash
cd Desktop
git clone https://github.com/ishan-nag/smart-resume-analyzer.git
cd smart-resume-analyzer
```

Your folder structure on Desktop will look like:

```
Desktop/
└── smart-resume-analyzer/
    ├── project-ai/
    ├── project-backend/
    └── project-frontend/
```

---

### Step 2 — Work inside your own folder only

Each teammate works only in their own subfolder:

| Teammate | Your folder |
|---|---|
| AI/ML | `project-ai/` |
| Backend | `project-backend/` |
| Frontend | `project-frontend/` |

Do NOT make changes in another teammate's folder.

---

### Step 3 — Create your own branch before making changes

Never push directly to `main`. Always work on your own branch.

```bash
# Create and switch to your branch
git checkout -b your-name/feature-name

# Examples:
git checkout -b backend/resume-upload-api
git checkout -b frontend/role-selection-page
git checkout -b ai/resume-analyzer-module
```

---

### Step 4 — Make your changes, then commit

After making changes inside your folder:

```bash
# Check what files you changed
git status

# Stage your changes
git add .

# Commit with a clear message
git commit -m "Add resume upload endpoint to backend"
```

---

### Step 5 — Push your branch to GitHub

```bash
git push origin your-branch-name

# Example:
git push origin backend/resume-upload-api
```

---

### Step 6 — Open a Pull Request (PR) to main

1. Go to the repository on GitHub: https://github.com/ishan-nag/smart-resume-analyzer
2. You will see a prompt: **"Compare & pull request"** — click it
3. Set the base branch to `main` and the compare branch to your branch
4. Write a short title and description of what you changed
5. Click **"Create pull request"**
6. The repo host (Ishan) will review and merge it

> Do NOT merge your own PR. Wait for the host to review and approve it.

---

### Step 7 — Keep your local repo up to date

Before starting work each day, pull the latest changes from main:

```bash
git checkout main
git pull origin main

# Switch back to your branch and bring in the latest main changes
git checkout your-branch-name
git merge main
```

If there are merge conflicts, resolve them in your editor, then:

```bash
git add .
git commit -m "Resolve merge conflicts"
```

---

## AI Module Setup (for AI/ML teammate only)

### Requirements

- Python 3.10 or higher
- pip

### Installation

```powershell
# Go into the AI module folder
cd Desktop\smart-resume-analyzer\project-ai

# Create a virtual environment
python -m venv .venv

# Activate it (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate it (Mac/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up your API key
copy .env.example .env
```

Open `.env` and add your Groq API key:

```
GROQ_API_KEY=your_key_here
```

Get a free key at https://console.groq.com

> Never push `.env` to GitHub. It is already in `.gitignore`.

---

## Running the AI Modules

Always run from the `project-ai/` folder, never from inside a subfolder.

```powershell
# Run in this order:
python -m resume_parser.resume_parser
python -m job_roles.job_roles
python -m ats_scorer.ats_scorer
python -m resume_analyzer.resume_analyzer
```

Run `resume_parser` first — it generates `output/parsed_resume.json` which other modules use.

---

## Handling Scanned / Image-Based PDFs

The AI module only supports text-based PDFs. If a candidate uploads a scanned or image-based PDF, the parser returns:

```json
{"error": "Could not extract text. File may be scanned or image-based."}
```

**Backend:** Check for the `error` key → return HTTP 400 with the error message.

**Frontend:** Show this message to the user:

```
"We couldn't read your resume. This usually means your PDF is image-based
or scanned. Please convert it to a text-based PDF and try again."
```

Suggest these free tools:
- https://www.smallpdf.com
- https://www.ilovepdf.com
- https://online2pdf.com

---

## For the Backend Developer

Every AI function returns a plain Python dict. Serialize it with `json.dumps()` and send it to the frontend.

The backend integration flow is:

```
1. parse_resume(pdf_path)                        → parsed_resume       [1 LLM call]
2. get_all_roles()                               → roles list          [0 LLM calls]
3. for each role_id: analyze_resume(parsed_resume, role_id)
                                                 → role_result         [1 LLM call each]
4. generate_upgrade_tip(all_role_results, parsed_resume)
                                                 → upgrade_tip         [1 LLM call]
5. best_match = max(all_role_results,
       key=lambda r: r["ats"]["overall_score"])  → best match role     [0 LLM calls]
6. Return full JSON to frontend
```

---

### 1. Resume Parser

Parses a PDF resume and returns structured candidate data.

```python
from resume_parser import parse_resume

result = parse_resume("uploads/resume.pdf")
```

Returns:
```json
{
    "name":       "John Doe",
    "email":      "john@gmail.com",
    "phone":      "+91 9876543210",
    "linkedin":   "linkedin.com/in/johndoe",
    "github":     "github.com/johndoe",
    "skills":     ["python", "docker", "react"],
    "education":  "B.Tech Computer Science, XYZ University, 2024",
    "experience": "Software Intern at ABC Corp, June–Aug 2023",
    "summary":    "Software engineer with 2 years of experience...",
    "raw_text":   "full resume text..."
}
```

On failure: `{"error": "File not found: uploads/resume.pdf"}`

**API calls: 1**

---

### 2. Job Roles

Returns the list of 28 supported job roles. Zero API calls — reads from a local JSON file.

```python
from job_roles import get_all_roles, get_role_by_id, build_job_description

# Get lightweight list for frontend dropdown
roles = get_all_roles()
```

Returns:
```json
[
    {"id": "ml_engineer",  "title": "Machine Learning Engineer", "category": "Data & AI",          "experience_level": "Mid-level"},
    {"id": "backend_engineer", "title": "Backend Engineer",      "category": "Software Engineering", "experience_level": "Mid-level"}
]
```

28 roles across 7 categories: Software Engineering, Data & AI, Infrastructure & Cloud, Mobile, Security, Product & Management, Emerging & Specialist.

**API calls: 0**

---

### 3. Resume Analyzer — PRIMARY MODULE

Analyzes a resume against a specific job role. Call once per role. The backend loops over selected roles.

```python
from resume_analyzer import analyze_resume, generate_upgrade_tip

# Call once per role
result = analyze_resume(parsed_resume, role_id="ml_engineer")
```

Returns:
```json
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
            "semantic_match":   {"score": 80, "feedback": "Strong backend experience but lacks cloud skills."},
            "experience_match": {"score": 70, "feedback": "2 years relevant experience, role requires 3-5."},
            "education_match":  {"score": 85, "feedback": "B.Tech Computer Science matches the requirement."}
        }
    },
    "skills_gap": {
        "matched":              ["python", "pytorch", "docker"],
        "missing":              ["tensorflow", "mlflow"],
        "nice_to_have_missing": ["kubernetes", "airflow"]
    },
    "quality_score": {
        "overall": 72,
        "breakdown": {"format": 75, "clarity": 70, "impact": 68, "brevity": 80}
    },
    "section_feedback": {
        "experience": {"score": 75, "feedback": "Good range of projects but lacks metrics.", "improvements": ["Add quantifiable achievements"]},
        "education":  {"score": 90, "feedback": "Degree is well-aligned.", "improvements": []},
        "summary":    {"score": 60, "feedback": "Too generic.", "improvements": ["Tailor to ML roles", "Mention top tools"]},
        "skills":     {"score": 80, "feedback": "Strong core skills listed.", "improvements": ["Add MLflow"]}
    }
}
```

On failure: `{"error": "Reason for failure"}`

**API calls: 1 per role**

---

### 4. Upgrade Tip Generator

Call ONCE after all `analyze_resume()` calls are done. Takes all role results and returns a single upgrade tip paragraph.

```python
# After looping over all roles:
tip = generate_upgrade_tip(all_role_results, parsed_resume)
```

Returns:
```json
{
    "upgrade_tip": "Your resume shows strong Python fundamentals across all roles
                    but consistently lacks cloud and deployment skills like Docker
                    and Kubernetes. Adding these with concrete project examples
                    would significantly improve your ATS scores."
}
```

**API calls: 1 total (called once, not per role)**

---

### 5. ATS Scorer (standalone)

Scores a resume against a role. Already used internally by `analyze_resume()` — backend does not need to call this separately unless needed standalone.

```python
from ats_scorer import score_resume

# New way — pass role_id
result = score_resume(parsed_resume, role_id="ml_engineer")

# Old way — pass raw job description string (still works)
result = score_resume(parsed_resume, job_description="We are looking for...")
```

**API calls: 1**

---

## API Call Budget

| Step | Module | Calls | Frequency |
|---|---|---|---|
| Parse resume | resume_parser | 1 | Once per session |
| Get all roles | job_roles | 0 | Once per session |
| Analyze per role | resume_analyzer | 1 per role | Per role selected |
| Global upgrade tip | resume_analyzer | 1 | Once after all roles |
| **Total (4 roles)** | | **6** | |

**Groq Free Tier limits (llama-3.3-70b-versatile):**

| Limit | Value |
|---|---|
| Requests per day | 1,000 |
| Requests per minute | 30 |
| Tokens per day | 100,000 |

At 6 calls per session, the free tier supports ~166 full sessions per day comfortably.

---

## Frontend → AI Output Mapping

| UI Element | Source |
|---|---|
| Best Match role title | `role.title` of max `ats.overall_score` across all roles |
| Resume Score number | `ats.overall_score` of best match role |
| Roles Compared count | `len(selected_role_ids)` — backend counts |
| Role card score % | `ats.overall_score` per role |
| Role card skill tags | `role.required_skills[:3]` from job_roles |
| ATS breakdown bars | `ats.breakdown` per role |
| Recommendation paragraph | `ats.recommendation` per role |
| Strengths bullets | `section_feedback[section].feedback` (positive sections) |
| Needs Improvement bullets | `section_feedback[section].improvements` |
| Suggested Resume Upgrade | `upgrade_tip.upgrade_tip` |
| Candidate Name | `parsed_resume.name` |
| Primary Stack | `", ".join(parsed_resume["skills"][:3])` — backend derives this |

---

## Project Folder Structure

```
project-ai/
├── ats_scorer/
│   ├── __init__.py
│   ├── ats_scorer.py
│   └── prompt_templates.py
├── data/
│   ├── sample_resume_1.pdf
│   ├── sample_resume_2.pdf
│   ├── sample_resume_3.pdf
│   ├── sample_resume_4.pdf
│   ├── sample_resume_5.pdf
│   ├── skills_list.json
│   └── job_roles.json
├── job_roles/
│   ├── __init__.py
│   └── job_roles.py
├── output/                   ← auto-generated, gitignored
├── resume_analyzer/
│   ├── __init__.py
│   ├── prompt_templates.py
│   └── resume_analyzer.py
├── resume_parser/
│   ├── __init__.py
│   ├── resume_parser.py
│   └── utils.py
├── shared/
│   ├── __init__.py
│   ├── groq_client.py
│   └── retry_handler.py
├── .env                      ← your API key, never push this
├── .env.example              ← safe to push, no real key
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Common Errors

| Error | Fix |
|---|---|
| `GROQ_API_KEY not found` | Check `.env` file — no spaces around `=` |
| `ModuleNotFoundError` | Run from `project-ai/` root, not a subfolder |
| `No module named 'groq'` | Activate venv first — `.\.venv\Scripts\Activate.ps1` |
| `model decommissioned` | Update `DEFAULT_MODEL` in `shared/groq_client.py` |
| `Could not extract text` | PDF must be text-based, not a scanned image |
| `git push rejected` | Run `git pull origin main --rebase` then push again |
| `pip installs to AppData` | Use `.\.venv\Scripts\python.exe -m pip install` instead |

---

## Data Privacy

This AI module is completely stateless between sessions. There is no accumulated history passed to the LLM — every API call generates a fresh prompt with only the current candidate's data. The LLM has zero knowledge of any previous resumes or candidates.

**Important for backend:** The `output/` JSON files are overwritten on every run. Always process a fresh upload for each candidate and never read leftover files from a previous session.

---

## Deployment Notes

At deployment time, the AI module will run as a separate Python service on Render with a FastAPI wrapper (`main.py`). The Spring Boot backend will call it via HTTP.

```
Frontend (Vercel) → Backend (Render, Java) → AI Module (Render, Python) → Groq API
```

> Render free tier sleeps after 15 minutes of inactivity. Open the app at least 1 minute before your demo.

The FastAPI `main.py` will be added at deployment time — it is not part of the current module.

---

**GitHub:** https://github.com/ishan-nag/smart-resume-analyzer  
**Last Updated:** Session 4 complete — resume analyzer module done.