"""
resume_parser/__init__.py
==========================
Exposes the public API of the resume_parser module.

Backend developers only need to import from here:

    from resume_parser import parse_resume, save_parsed_resume

Available functions:
    - parse_resume(pdf_path, skills_file_path)  → dict
    - save_parsed_resume(parsed_data, output_path) → None
"""
from resume_parser.resume_parser import parse_resume, save_parsed_resume

__all__ = ["parse_resume", "save_parsed_resume"]