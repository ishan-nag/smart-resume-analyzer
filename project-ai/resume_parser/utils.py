"""
utils.py — Resume Parser Helper Functions
==========================================
This module provides lightweight utility functions used by resume_parser.py
to extract contact information from raw resume text using regex.

These fields use regex because they follow universal, predictable formats:
    - Email address    → always follows user@domain.com pattern
    - Phone number     → always a sequence of digits with separators
    - LinkedIn URL     → always contains linkedin.com/in/
    - GitHub URL       → always contains github.com/

All other fields (name, skills, education, experience) are extracted
by the Groq LLM in resume_parser.py for maximum accuracy across
all resume formats and layouts.

For backend integration:
    - All functions accept plain text (str) as input
    - All functions return str
"""

import re


# Email Extraction

def extract_email(text: str) -> str:
    """
    Extracts the first email address found in the resume text.

    Parameters:
        text (str): Raw text extracted from the resume.

    Returns:
        str: The first email address found, or an empty string if none found.

    Example:
        extract_email("Contact me at john.doe@gmail.com")
        # Returns: "john.doe@gmail.com"
    """
    pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    match = re.search(pattern, text)
    return match.group(0) if match else ""


# Phone Number Extraction

def extract_phone(text: str) -> str:
    """
    Extracts the first phone number found in the resume text.
    Supports formats like:
        +1-800-555-0199, (800) 555-0199, 8005550199,
        +91 9876543210, (480) 123-5689

    Parameters:
        text (str): Raw text extracted from the resume.

    Returns:
        str: The first phone number found, or an empty string if none found.

    Example:
        extract_phone("San Francisco | (480) 123-5689 | email@gmail.com")
        # Returns: "(480) 123-5689"
    """
    pattern = r'(\+?\d{1,3}[\s\-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}'
    match = re.search(pattern, text)
    return match.group(0).strip() if match else ""


# LinkedIn & GitHub Extraction

def extract_linkedin(text: str) -> str:
    """
    Extracts a LinkedIn profile URL from the resume text.

    Parameters:
        text (str): Raw text extracted from the resume.

    Returns:
        str: LinkedIn URL if found, otherwise an empty string.

    Example:
        extract_linkedin("linkedin.com/in/johndoe")
        # Returns: "linkedin.com/in/johndoe"
    """
    pattern = r'(https?://)?(www\.)?linkedin\.com/in/[a-zA-Z0-9\-_/]+'
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(0).strip() if match else ""


def extract_github(text: str) -> str:
    """
    Extracts a GitHub profile URL from the resume text.

    Parameters:
        text (str): Raw text extracted from the resume.

    Returns:
        str: GitHub URL if found, otherwise an empty string.

    Example:
        extract_github("github.com/johndoe")
        # Returns: "github.com/johndoe"
    """
    pattern = r'(https?://)?(www\.)?github\.com/[a-zA-Z0-9\-_/]+'
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(0).strip() if match else ""