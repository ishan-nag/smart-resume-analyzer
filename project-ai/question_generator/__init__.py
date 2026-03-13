"""
question_generator/__init__.py
================================
Exposes the public API of the question_generator module.

Backend developers only need to import from here:

    from question_generator import generate_questions, save_generated_questions

Available functions:
    - generate_questions(parsed_resume, job_description, num_questions) → dict
    - save_generated_questions(questions, output_path) → None

Input:
    parsed_resume   (dict) — output of resume_parser's parse_resume()
    job_description (str)  — full job description text
    num_questions   (int)  — questions per type, default is 5

Output:
    {
        "technical":    [5 questions],
        "behavioral":   [5 questions],
        "hr_general":   [5 questions],
        "resume_based": [5 questions]
    }
"""

from question_generator.question_generator import generate_questions, save_generated_questions

__all__ = ["generate_questions", "save_generated_questions"]