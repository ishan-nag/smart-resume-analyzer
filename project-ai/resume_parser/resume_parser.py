"""
resume_parser.py — Main Resume Parser Module
=============================================
This module parses a PDF resume and extracts structured information
using a two-step approach:

    Step 1 — Regex (utils.py):
        Extracts contact fields that follow universal formats:
        email, phone, linkedin, github

    Step 2 — Groq LLM:
        Extracts fields that vary by resume format and layout:
        name, skills, education, experience, summary

        Using LLM here makes the parser robust to ANY resume format:
        - Spaced headers (E D U C A T I O N)
        - Two-column layouts
        - Non-standard section names (Work History, Background, etc.)
        - International resumes
        - Creative/design resumes

Output format:
    A Python dictionary (easily serializable to JSON):
    {
        "name":        str,
        "email":       str,
        "phone":       str,
        "linkedin":    str,
        "github":      str,
        "skills":      list[str],
        "education":   str,
        "experience":  str,
        "summary":     str,
        "raw_text":    str
    }

For backend integration (Java Spring Boot):
    - Call parse_resume(pdf_path) with the path to the uploaded PDF
    - Returns a dict → serialize with json.dumps() to send as JSON
    - Optionally call save_parsed_resume(result, output_path) to persist

Dependencies:
    pip install groq python-dotenv pdfplumber
"""

import os
import json
import pdfplumber

# ── Regex-based contact extractors ──
from resume_parser.utils import (
    extract_email,
    extract_phone,
    extract_linkedin,
    extract_github,
)

# ── Shared Groq client and retry handler ──
from shared.groq_client import get_groq_client, MODEL_CONFIGS
from shared.retry_handler import call_with_retry, parse_json_response


# PDF Text Extraction

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts raw plain text from a PDF file using pdfplumber.

    Parameters:
        pdf_path (str): Path to the PDF resume file.

    Returns:
        str: All text extracted from every page, joined by newlines.
             Returns an empty string if extraction fails.

    Example:
        text = extract_text_from_pdf("data/sample_resume.pdf")
    """
    all_text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    all_text.append(page_text)
    except Exception as e:
        print(f"[ResumeParser] ERROR: Could not read PDF '{pdf_path}'. Reason: {e}")
        return ""
    return '\n'.join(all_text)


# LLM-Based Field Extraction

def extract_fields_with_llm(raw_text: str) -> dict:
    """
    Uses Groq LLM to extract structured fields from raw resume text.
    This handles any resume format, layout, or style generically.

    Fields extracted by LLM:
        - name       : candidate's full name
        - skills     : list of technical and soft skills
        - education  : education background summary
        - experience : work experience summary
        - summary    : professional summary or objective

    Parameters:
        raw_text (str): Raw text extracted from the PDF.

    Returns:
        dict: Extracted fields as a Python dict.
              Returns empty strings/lists on failure.
    """

    prompt = f"""You are an expert resume parser.

Extract the following fields from the resume text below and return them as a JSON object.

RESUME TEXT:
{raw_text}

Extract these fields:
- name        : The candidate's full name (string)
- skills      : A list of all technical skills, tools, frameworks, and programming languages mentioned (list of strings, all lowercase)
- education   : Full education background including degrees, institutions, and years (string)
- experience  : Full work experience including company names, roles, and responsibilities (string)
- summary     : Professional summary or objective statement if present, otherwise empty string (string)

Rules:
- For skills, extract ALL technologies, languages, frameworks, tools mentioned anywhere in the resume
- For education and experience, preserve the full content of those sections
- If a field is not found, use an empty string "" or empty list []
- Do NOT summarize or shorten education or experience — return the full text of those sections

Respond ONLY with a valid JSON object. No explanation, no markdown, no extra text.
The JSON must have exactly these keys: name, skills, education, experience, summary
"""

    try:
        client = get_groq_client()
    except ValueError as e:
        print(f"[ResumeParser] {e}")
        return _empty_llm_fields()

    config = MODEL_CONFIGS["question_generator"]  # reuse similar config

    raw_response = call_with_retry(
        client       = client,
        messages     = [
            {
                "role": "system",
                "content": (
                    "You are an expert resume parser. "
                    "Always respond with ONLY a valid JSON object. "
                    "No explanation, no markdown, no extra text."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        model        = config["model"],
        temperature  = 0.1,     # Very low — we want consistent, factual extraction
        max_tokens   = 2048,    # Resumes can be long
        caller_label = "ResumeParser:LLM",
    )

    result = parse_json_response(raw_response, "ResumeParser:LLM")

    if result is None or not isinstance(result, dict):
        print("[ResumeParser] WARNING: LLM extraction failed. Returning empty fields.")
        return _empty_llm_fields()

    # ── Sanitize and return ──
    return {
        "name":       str(result.get("name", "")).strip(),
        "skills":     [s.lower().strip() for s in result.get("skills", []) if s],
        "education":  str(result.get("education", "")).strip(),
        "experience": str(result.get("experience", "")).strip(),
        "summary":    str(result.get("summary", "")).strip(),
    }


def _empty_llm_fields() -> dict:
    """Returns a safe empty dict when LLM extraction fails."""
    return {
        "name":       "",
        "skills":     [],
        "education":  "",
        "experience": "",
        "summary":    "",
    }


#Main Parse Function

def parse_resume(pdf_path: str) -> dict:
    """
    Main function to parse a resume PDF and extract structured information.
    This is the PRIMARY function the backend should call.

    Uses a two-step approach:
        1. Regex   → extracts email, phone, linkedin, github (fast, no API call)
        2. Groq LLM → extracts name, skills, education, experience, summary
                      (robust, works on any resume format)

    Parameters:
        pdf_path (str):
            Path to the uploaded PDF resume file.
            Example: "data/sample_resume.pdf" or "/uploads/resume_john.pdf"

    Returns:
        dict: A structured dictionary containing all extracted resume fields.

        Example return value:
        {
            "name":       "John Doe",
            "email":      "john.doe@gmail.com",
            "phone":      "+91 9876543210",
            "linkedin":   "linkedin.com/in/johndoe",
            "github":     "github.com/johndoe",
            "skills":     ["docker", "java", "python", "spring boot"],
            "education":  "B.Tech in Computer Science, XYZ University, 2024",
            "experience": "Software Intern at ABC Corp, June 2023 – Aug 2023",
            "summary":    "Results-driven software engineer with 2 years of experience...",
            "raw_text":   "John Doe\njohn.doe@gmail.com\n..."
        }

        On failure, returns:
        {
            "error": "Reason for failure"
        }

    Example usage (Python):
        from resume_parser.resume_parser import parse_resume

        result = parse_resume("data/sample_resume.pdf")
        print(result["skills"])    # ["python", "react", ...]
        print(result["email"])     # "john@example.com"
    """

    # ── Step 1: Validate file ──
    if not os.path.exists(pdf_path):
        return {"error": f"File not found: {pdf_path}"}

    # ── Step 2: Extract raw text from PDF ──
    print(f"[ResumeParser] Extracting text from: {pdf_path}")
    raw_text = extract_text_from_pdf(pdf_path)

    if not raw_text.strip():
        return {"error": "Could not extract text. File may be scanned or image-based."}

    # ── Step 3: Regex extraction (contact info) ──
    print("[ResumeParser] Extracting contact info via regex...")
    email    = extract_email(raw_text)
    phone    = extract_phone(raw_text)
    linkedin = extract_linkedin(raw_text)
    github   = extract_github(raw_text)

    # ── Step 4: LLM extraction (all other fields) ──
    print("[ResumeParser] Extracting resume fields via Groq LLM...")
    llm_fields = extract_fields_with_llm(raw_text)

    # ── Step 5: Combine all fields ──
    parsed_data = {
        "name":       llm_fields["name"],
        "email":      email,
        "phone":      phone,
        "linkedin":   linkedin,
        "github":     github,
        "skills":     llm_fields["skills"],
        "education":  llm_fields["education"],
        "experience": llm_fields["experience"],
        "summary":    llm_fields["summary"],
        "raw_text":   raw_text,
    }

    print("[ResumeParser] Parsing complete!")
    return parsed_data


# Save Output to JSON

def save_parsed_resume(parsed_data: dict, output_path: str = "output/parsed_resume.json") -> None:
    """
    Saves the parsed resume dictionary to a JSON file.

    Parameters:
        parsed_data (dict): The structured resume data returned by parse_resume().
        output_path (str):  Path where the JSON file should be saved.
                            Defaults to "output/parsed_resume.json"

    Returns:
        None

    Example usage:
        result = parse_resume("data/sample_resume.pdf")
        save_parsed_resume(result, "output/parsed_resume.json")
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed_data, f, indent=4, ensure_ascii=False)
    print(f"[ResumeParser] Parsed resume saved to: {output_path}")


# Test

if __name__ == "__main__":
    """
    Quick test — run this file directly to verify the parser works.

    Usage (from project root):
        python -m resume_parser.resume_parser

    Make sure:
        - .env file has GROQ_API_KEY set
        - data/ folder has at least one sample_resume*.pdf file
    """
    import glob
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # ── Auto-detect all PDFs in data/ folder ──
    pdf_files = sorted(glob.glob("data/sample_resume*.pdf"))

    if not pdf_files:
        print("[ResumeParser] No PDF files found in data/ folder.")
        print("Add a resume PDF named sample_resume.pdf or sample_resume_1.pdf etc.")
        exit(1)

    print(f"[ResumeParser] Found {len(pdf_files)} resume(s): {pdf_files}\n")

    for pdf_path in pdf_files:
        print(f"\n{'='*60}")
        print(f"[ResumeParser] Parsing: {pdf_path}")
        print('='*60)

        result = parse_resume(pdf_path)

        if "error" in result:
            print(f"[ResumeParser] FAILED: {result['error']}")
        else:
            print(f"\n  name       : {result['name']}")
            print(f"  email      : {result['email']}")
            print(f"  phone      : {result['phone']}")
            print(f"  linkedin   : {result['linkedin']}")
            print(f"  github     : {result['github']}")
            print(f"  skills     : {result['skills']}")
            print(f"  summary    : {result['summary'][:100]}..." if result['summary'] else "  summary    : ")
            print(f"  education  : {result['education'][:150]}..." if result['education'] else "  education  : ")
            print(f"  experience : {result['experience'][:150]}..." if result['experience'] else "  experience : ")
            print(f"  raw_text   : [truncated, {len(result['raw_text'])} characters]")

            # Save with matching filename
            base_name   = os.path.splitext(os.path.basename(pdf_path))[0]
            output_name = base_name.replace("sample_resume", "parsed_resume")
            output_path = f"output/{output_name}.json"
            save_parsed_resume(result, output_path)