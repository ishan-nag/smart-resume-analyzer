# Smart Mock Interview Tool — AI Module

The AI module handles four things: parsing resumes, generating interview questions, scoring resumes against job descriptions, and evaluating candidate answers. It's written in Python and uses the Groq API for LLM calls.

---

## Team

- **AI/ML** — This module (Python)
- **Backend** — Java Spring Boot, calls functions from this module
- **Frontend** — Talks to the backend, never touches this module directly

---

## Setup

You need Python 3.10+ and pip installed.

```bash
# 1. Go into the project folder
cd project-ai

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1
# Mac/Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your API key
copy .env.example .env       # Windows
cp .env.example .env         # Mac/Linux
```

Open `.env` and add your Groq API key:
```
GROQ_API_KEY=your_key_here
```

Get a free key at https://console.groq.com

> Never push `.env` to GitHub. It's already in `.gitignore`.

---

## Running the modules

Always run from the `project-ai/` folder, not from inside any subfolder.

```bash
python -m resume_parser.resume_parser
python -m question_generator.question_generator
python -m ats_scorer.ats_scorer
python -m answer_evaluator.answer_evaluator
python -m shared.retry_handler
```

Run `resume_parser` first — other modules use `output/parsed_resume.json` that it generates.

---

## Handling Scanned / Image-Based PDFs

The AI module only supports text-based PDFs. If a candidate uploads a scanned
or image-based PDF, the parser returns this response:

```json
{"error": "Could not extract text. File may be scanned or image-based."}
```

**Backend:** Check for the `error` key in the response and return a 400 status to the frontend with the error message.

**Frontend:** Show this message to the user:

```
"We couldn't read your resume. This usually means your PDF is image-based
or scanned. Please convert it to a text-based PDF and try again."
```

You can also suggest these free conversion websites:
- https://www.smallpdf.com
- https://www.ilovepdf.com
- https://online2pdf.com

These sites convert scanned PDFs into text-based PDFs for free in under a minute.
The candidate converts their file, re-uploads, and everything works normally.

---

## For the backend developer

Every function returns a plain Python dict. Serialize it with `json.dumps()` and send it to the frontend.

---

### 1. Resume Parser

Parses a PDF resume and returns structured candidate data. Works on any resume format — single or multi-page, any layout.

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
    "skills":     ["python", "docker", "react", "spring boot"],
    "education":  "B.Tech Computer Science, XYZ University, 2024",
    "experience": "Software Intern at ABC Corp, June–Aug 2023",
    "summary":    "Software engineer with 2 years of experience...",
    "raw_text":   "full resume text..."
}
```

If it fails: `{"error": "File not found: uploads/resume.pdf"}`

Pass the full result dict to all other modules. The `raw_text` field inside it is used by the question generator and ATS scorer.

**API calls: 1**

---

### 2. Question Generator

Generates 20 interview questions (5 per type) based on the resume and job description in a single API call.

```python
from question_generator import generate_questions

questions = generate_questions(parsed_resume, job_description)
```

`parsed_resume` is the dict from `parse_resume()`.
`job_description` is a plain string.

Returns:
```json
{
    "technical":    ["Question 1", "Question 2", "..."],
    "behavioral":   ["Question 1", "Question 2", "..."],
    "hr_general":   ["Question 1", "Question 2", "..."],
    "resume_based": ["Question 1", "Question 2", "..."]
}
```

**API calls: 1**

---

### 3. ATS Scorer

Scores how well the resume matches the job description. Returns a score out of 100 with a breakdown. All three LLM scores are fetched in a single API call.

```python
from ats_scorer import score_resume

result = score_resume(parsed_resume, job_description)
```

Returns:
```json
{
    "overall_score":  76.8,
    "recommendation": "Good Match",
    "breakdown": {
        "keyword_match": {
            "score": 53,
            "matched_keywords": ["docker", "java", "python"],
            "missing_keywords": ["kubernetes", "aws"],
            "feedback": "Matches 7 out of 13 skills."
        },
        "semantic_match":   {"score": 92, "feedback": "Strong overall fit."},
        "experience_match": {"score": 90, "feedback": "Experience aligns well."},
        "education_match":  {"score": 80, "feedback": "Degree matches requirement."}
    }
}
```

Recommendation ranges: Excellent Match (85+), Good Match (70–84), Moderate Match (55–69), Weak Match (40–54), Poor Match (below 40).

**API calls: 1**

---

### 4. Answer Evaluator

Evaluates candidate answers on four criteria — relevance, technical accuracy, clarity, and completeness — and returns an ideal reference answer. In session mode, answers are evaluated in batches of 5 to minimize API calls.

**Single answer:**
```python
from answer_evaluator import evaluate_answer

result = evaluate_answer(
    question         = "What is Docker?",
    candidate_answer = "Docker is a containerization tool...",
    question_type    = "technical"   # technical / behavioral / hr_general / resume_based
)
```

Returns:
```json
{
    "question":         "What is Docker?",
    "question_type":    "technical",
    "candidate_answer": "Docker is a containerization tool...",
    "overall_score":    82.5,
    "performance":      "Good",
    "breakdown": {
        "relevance":          {"score": 85, "feedback": "Directly answers the question."},
        "technical_accuracy": {"score": 80, "feedback": "Correct but incomplete."},
        "clarity":            {"score": 85, "feedback": "Well structured."},
        "completeness":       {"score": 75, "feedback": "Missing some use cases."}
    },
    "ideal_answer": "Docker is an open-source platform that..."
}
```

Performance ranges: Excellent (85+), Good (70–84), Average (55–69), Needs Improvement (40–54), Poor (below 40).

**Full session:**
```python
from answer_evaluator import evaluate_session

qa_pairs = [
    {"question": "What is Docker?",        "answer": "...", "question_type": "technical"},
    {"question": "Tell me about yourself.", "answer": "...", "question_type": "hr_general"},
]

report = evaluate_session(qa_pairs)
```

Returns:
```json
{
    "total_questions":  2,
    "average_score":    79.5,
    "overall_feedback": "Strong technical answers, HR responses need more structure.",
    "strengths":        ["Technical depth", "Clear explanations"],
    "improvements":     ["Use STAR method for behavioral questions"],
    "evaluations":      ["...individual result per question..."]
}
```

**API calls: 1 per 5 answers (batched) + 1 for the summary**

---

## Full session flow

```python
from resume_parser import parse_resume
from question_generator import generate_questions
from ats_scorer import score_resume
from answer_evaluator import evaluate_session

resume    = parse_resume("uploads/resume.pdf")
job_desc  = "Looking for a backend engineer with Java and Spring Boot..."

ats       = score_resume(resume, job_desc)
questions = generate_questions(resume, job_desc)

qa_pairs  = [
    {"question": questions["technical"][0],  "answer": "...", "question_type": "technical"},
    {"question": questions["behavioral"][0], "answer": "...", "question_type": "behavioral"},
]

report = evaluate_session(qa_pairs)
```

---

## API usage

| Module | Calls |
|---|---|
| resume_parser | 1 |
| ats_scorer | 1 |
| question_generator | 1 |
| answer_evaluator | 1 per 5 answers + 1 summary |
| **Total (20 questions)** | **~8 per session** |

Groq free tier limits for llama-3.3-70b-versatile:

| Limit | Value |
|---|---|
| Requests per day | 1,000 |
| Requests per minute | 30 |
| Tokens per day | 100,000 |

At ~8 calls and ~8,000 tokens per session, the free tier comfortably supports around 12 full sessions per day. For a college demo this is more than enough.

---

## Common errors

| Error | Fix |
|---|---|
| `GROQ_API_KEY not found` | Check `.env` file — no spaces around `=` |
| `File not found` | Run commands from `project-ai/` not a subfolder |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `model decommissioned` | Update `DEFAULT_MODEL` in `shared/groq_client.py` |
| `Could not extract text` | PDF must be text-based, not a scanned image |

---

## Backend Integration Examples

### Subprocess (Local Development)

```java
@PostMapping("/parse-resume")
public ResponseEntity<?> parseResume(@RequestParam("file") MultipartFile file) {
    File tempFile = File.createTempFile("resume", ".pdf");
    file.transferTo(tempFile);
    
    ProcessBuilder pb = new ProcessBuilder(
        "python", "-c",
        "import sys; sys.path.insert(0, '../project-ai'); " +
        "from resume_parser import parse_resume; " +
        "import json; " +
        "result = parse_resume('" + tempFile.getAbsolutePath() + "'); " +
        "print(json.dumps(result))"
    );
    
    Process process = pb.start();
    BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
    String jsonOutput = reader.lines().collect(Collectors.joining());
    
    ObjectMapper mapper = new ObjectMapper();
    return ResponseEntity.ok(mapper.readValue(jsonOutput, Object.class));
}
```

### HTTP (Deployed on Render)

```java
@Service
public class AIService {
    private static final String AI_API_URL = "https://mock-interview-ai.onrender.com";
    
    public ResponseEntity<?> parseResume(String filePath) {
        RestTemplate restTemplate = new RestTemplate();
        String url = AI_API_URL + "/parse-resume?path=" + filePath;
        return restTemplate.getForEntity(url, Object.class);
    }
}
```

### Error Handling

```java
if (response.get("error") != null) {
    if (response.get("error").contains("scanned")) {
        return "Please convert your PDF to text-based format";
    }
    return response.get("error");
}
```

---

## Project Structure for the AI part

```
project-ai/
├── resume_parser/
│   ├── __init__.py
│   ├── resume_parser.py
│   └── utils.py
├── question_generator/
│   ├── __init__.py
│   ├── question_generator.py
│   └── prompt_templates.py
├── ats_scorer/
│   ├── __init__.py
│   ├── ats_scorer.py
│   └── prompt_templates.py
├── answer_evaluator/
│   ├── __init__.py
│   ├── answer_evaluator.py
│   └── prompt_templates.py
├── shared/
│   ├── __init__.py
│   ├── groq_client.py
│   └── retry_handler.py
├── data/
│   ├── sample_resume.pdf
│   ├── sample_job_description.txt
│   └── skills_list.json
├── output/
│   ├── parsed_resume.json
│   ├── ats_result.json
│   ├── generated_questions.json
│   └── evaluation_result.json
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## For the Frontend Developer

- Expect JSON responses from the backend (which calls our Python functions)
- Check for `error` key in responses:
  - If present and contains "scanned" → show: "Please convert your PDF to text-based format"
  - If present otherwise → show the error message
  - If `error` is null → data is good, use the `result` key
- For PDF conversion: link to smallpdf.com, ilovepdf.com, or online2pdf.com

---

## For the Team Lead

**API Optimization:** Module uses ~8 API calls per full interview session (down from 29 originally).

**Capacity:** Groq free tier supports ~12 full sessions per day (token-limited at 100k/day).

**Deployment:** At production time, AI module gets a FastAPI wrapper for HTTP integration with Spring Boot backend.

**Error Handling:** All scanned PDF errors are handled gracefully with user-friendly messages.

---

## Data Privacy & Session Memory

**This AI module is completely stateless between sessions.** 
- There is **no accumulated chat history** passed to the LLM. Every API call generates a fresh prompt consisting only of the current candidate's data.
- The LLM has zero knowledge of any previous resumes, interviews, or candidates.
- This entirely prevents "cross-contamination" and hallucinations where the AI might accidentally ask a candidate about someone else's resume.

**Important for Backend:** 
The `output/` JSON files are silently overwritten on every run. Always ensure you process a fresh upload for each candidate, and never accidentally read leftover `output/parsed_resume.json` from a previous session. Calling `reset_groq_client()` from `shared.groq_client` at the end of a session is also good practice to fully clear the API connection.

---

## Deployment Notes

When deploying to Render, the AI module will run as a separate Python service with a FastAPI wrapper (`main.py`). The Spring Boot backend will call it via HTTP endpoints.

Create `main.py`:

```python
from fastapi import FastAPI
from resume_parser import parse_resume
from question_generator import generate_questions
from ats_scorer import score_resume
from answer_evaluator import evaluate_answer, evaluate_session

app = FastAPI()

@app.post("/parse-resume")
async def api_parse_resume(file_path: str):
    return parse_resume(file_path)

@app.post("/generate-questions")
async def api_generate_questions(resume: dict, job_description: str):
    return generate_questions(resume, job_description)

@app.post("/score-resume")
async def api_score_resume(resume: dict, job_description: str):
    return score_resume(resume, job_description)

@app.post("/evaluate-answer")
async def api_evaluate_answer(question: str, answer: str, question_type: str):
    return evaluate_answer(question, answer, question_type)

@app.post("/evaluate-session")
async def api_evaluate_session(qa_pairs: list):
    return evaluate_session(qa_pairs)

@app.get("/health")
async def health():
    return {"status": "ok"}
```

---

**GitHub:** https://github.com/ishan-nag/mock-interview-ai  
**Last Updated:** 2026

## How Teammates Should Add Their Modules

Follow these steps to add your project module (backend / frontend) to the repository.

### 1. Clone the repository

```bash
git clone <repo-url>
2. Go inside the project folder
cd "Mock Interview Project Main"
3. Create your module folder

Example:

mkdir project-backend
mkdir project-frontend

Your structure should look like this:

Mock Interview Project Main/
│
├── project-ai/
├── project-backend/
└── project-frontend/
4. Add your files inside your folder

Example:

project-backend/
│
├── src/
├── pom.xml
└── README.md
5. Commit and push the changes
git add .
git commit -m "Add backend module"
git push

After pushing, the repository will look like:

repo-root/
│
├── project-ai/
├── project-backend/
└── project-frontend/