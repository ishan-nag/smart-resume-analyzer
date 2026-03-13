"""
prompt_templates.py — Prompt Templates for Answer Evaluator
============================================================
This module contains all prompt strings used by answer_evaluator.py
to evaluate candidate interview answers via the Groq LLM API.

Keeping prompts separate from logic makes it easy to:
    - Tweak evaluation criteria without touching evaluator logic
    - Add new evaluation types in the future
    - Review and improve prompts independently

OPTIMIZATION NOTE:
    Added get_batch_evaluation_prompt() which evaluates multiple answers
    (up to 5) in a single API call instead of one call per answer.
    This reduces API calls significantly for full sessions.
"""


def get_evaluation_prompt(question: str, candidate_answer: str, question_type: str) -> str:
    """
    Returns a prompt to evaluate a single candidate's interview answer
    across 4 criteria and also generate an ideal reference answer.

    Used by evaluate_answer() for single answer evaluation.

    Parameters:
        question (str):          The interview question that was asked.
        candidate_answer (str):  The candidate's response to the question.
        question_type (str):     Type of question — one of:
                                 "technical", "behavioral", "hr_general", "resume_based"

    Returns:
        str: A formatted prompt string for the Groq API.

    Expected LLM response format (JSON):
        {
            "relevance":          {"score": 80, "feedback": "..."},
            "technical_accuracy": {"score": 75, "feedback": "..."},
            "clarity":            {"score": 85, "feedback": "..."},
            "completeness":       {"score": 70, "feedback": "..."},
            "ideal_answer":       "A complete ideal answer..."
        }
    """
    type_context = {
        "technical":    "This is a technical interview question for a systems/infrastructure role. Focus on technical accuracy, depth of reasoning, and whether the candidate demonstrates systems thinking and awareness of engineering tradeoffs at scale.",
        "behavioral":   "This is a behavioral question. Evaluate using the STAR method (Situation, Task, Action, Result).",
        "hr_general":   "This is an HR/general question. Focus on clarity, professionalism, and relevance.",
        "resume_based": "This is a resume-based question. Evaluate how well the candidate explained their own experience.",
    }
    context = type_context.get(question_type, "This is a general interview question.")

    return f"""You are an expert interview coach evaluating a candidate's answer.

{context}

INTERVIEW QUESTION:
{question}

CANDIDATE'S ANSWER:
{candidate_answer}

Evaluate on 4 criteria (0-100 each):
1. relevance          — Does it directly address what was asked?
2. technical_accuracy — Is the content factually correct and technically sound?
3. clarity            — Is it well-structured and easy to understand?
4. completeness       — Does it cover all important aspects?

Also provide an ideal reference answer.

Respond ONLY with a valid JSON object. No explanation, no markdown, no extra text.
{{
    "relevance":          {{"score": <0-100>, "feedback": "<one sentence>"}},
    "technical_accuracy": {{"score": <0-100>, "feedback": "<one sentence>"}},
    "clarity":            {{"score": <0-100>, "feedback": "<one sentence>"}},
    "completeness":       {{"score": <0-100>, "feedback": "<one sentence>"}},
    "ideal_answer":       "<complete model answer>"
}}
"""


def get_batch_evaluation_prompt(qa_batch: list) -> str:
    """
    Returns a prompt to evaluate multiple answers (up to 5) in a single API call.

    Used by evaluate_session() to batch answers instead of calling
    the API once per answer — significantly reduces total API calls.

    Parameters:
        qa_batch (list): A list of dicts, each with:
                         - "question"      (str)
                         - "answer"        (str)
                         - "question_type" (str)
                         Maximum 5 items per batch.

    Returns:
        str: A formatted prompt string for the Groq API.

    Expected LLM response format (JSON):
        {
            "evaluations": [
                {
                    "relevance":          {"score": 80, "feedback": "..."},
                    "technical_accuracy": {"score": 75, "feedback": "..."},
                    "clarity":            {"score": 85, "feedback": "..."},
                    "completeness":       {"score": 70, "feedback": "..."},
                    "ideal_answer":       "..."
                },
                ...
            ]
        }
    """
    type_context = {
        "technical":    "technical — focus on technical accuracy",
        "behavioral":   "behavioral — use STAR method evaluation",
        "hr_general":   "hr_general — focus on clarity and professionalism",
        "resume_based": "resume_based — evaluate explanation of own experience",
    }

    qa_text = ""
    for i, pair in enumerate(qa_batch, 1):
        qtype   = pair.get("question_type", "technical")
        context = type_context.get(qtype, "general")
        qa_text += f"""
--- Answer {i} ({context}) ---
QUESTION: {pair.get('question', '')}
CANDIDATE ANSWER: {pair.get('answer', '')}
"""

    return f"""You are an expert interview coach evaluating multiple candidate answers.

{qa_text}

For EACH answer above, evaluate on 4 criteria (0-100 each):
1. relevance          — Does it directly address what was asked?
2. technical_accuracy — Is the content factually correct and technically sound?
3. clarity            — Is it well-structured and easy to understand?
4. completeness       — Does it cover all important aspects?

Also provide an ideal reference answer for each.

Respond ONLY with a valid JSON object. No explanation, no markdown, no extra text.
Return evaluations in the same order as the answers above.
{{
    "evaluations": [
        {{
            "relevance":          {{"score": <0-100>, "feedback": "<one sentence>"}},
            "technical_accuracy": {{"score": <0-100>, "feedback": "<one sentence>"}},
            "clarity":            {{"score": <0-100>, "feedback": "<one sentence>"}},
            "completeness":       {{"score": <0-100>, "feedback": "<one sentence>"}},
            "ideal_answer":       "<complete model answer>"
        }},
        ... (one entry per answer)
    ]
}}
"""


def get_batch_summary_prompt(evaluations: list) -> str:
    """
    Returns a prompt to generate an overall performance summary
    after evaluating multiple answers in a session.

    Parameters:
        evaluations (list): A list of evaluation result dicts,
                            each returned by evaluate_answer().

    Returns:
        str: A formatted prompt string for the Groq API.

    Expected LLM response format (JSON):
        {
            "overall_feedback": "Strong technical skills but needs improvement on behavioral answers.",
            "strengths":        ["Technical knowledge", "Clear communication"],
            "improvements":     ["Use STAR method", "Be more concise"]
        }
    """
    summary_lines = []
    for i, ev in enumerate(evaluations, 1):
        summary_lines.append(
            f"Q{i} ({ev.get('question_type', 'general')}): "
            f"Overall={ev.get('overall_score', 'N/A')}, "
            f"Relevance={ev.get('breakdown', {}).get('relevance', {}).get('score', 'N/A')}, "
            f"Technical={ev.get('breakdown', {}).get('technical_accuracy', {}).get('score', 'N/A')}, "
            f"Clarity={ev.get('breakdown', {}).get('clarity', {}).get('score', 'N/A')}, "
            f"Completeness={ev.get('breakdown', {}).get('completeness', {}).get('score', 'N/A')}"
        )

    evaluations_text = "\n".join(summary_lines)

    return f"""You are an expert interview coach reviewing a candidate's overall mock interview performance.

Here is a summary of their scores across all questions:
{evaluations_text}

Provide:
1. A brief overall performance feedback (2-3 sentences)
2. Top 2-3 strengths demonstrated
3. Top 2-3 areas for improvement with specific advice

Respond ONLY with a valid JSON object. No explanation, no markdown, no extra text.
{{
    "overall_feedback": "<2-3 sentence summary>",
    "strengths":        ["<strength 1>", "<strength 2>", "<strength 3>"],
    "improvements":     ["<improvement 1>", "<improvement 2>", "<improvement 3>"]
}}
"""