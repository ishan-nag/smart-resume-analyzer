"""
answer_evaluator.py — Interview Answer Evaluator
=================================================
This module evaluates a candidate's answer to an interview question
using the Groq LLM API.

Each answer is evaluated on 4 criteria in BATCHES of 5 (saves API calls):
    1. Relevance          — Does the answer address what was asked?
    2. Technical Accuracy — Is the content factually correct?
    3. Clarity            — Is it well-structured and easy to understand?
    4. Completeness       — Does it cover all important aspects?

Additionally:
    - An ideal reference answer is provided for the candidate to learn from
    - A batch summary function evaluates overall session performance
      after multiple answers are evaluated

Output format for a single evaluation:
    {
        "question":         str,
        "question_type":    str,
        "candidate_answer": str,
        "overall_score":    float (0-100),
        "performance":      str,
        "breakdown": {
            "relevance":          {"score": int, "feedback": str},
            "technical_accuracy": {"score": int, "feedback": str},
            "clarity":            {"score": int, "feedback": str},
            "completeness":       {"score": int, "feedback": str}
        },
        "ideal_answer": str
    }

Output format for a session summary (multiple answers):
    {
        "total_questions":  int,
        "average_score":    float,
        "overall_feedback": str,
        "strengths":        list[str],
        "improvements":     list[str],
        "evaluations":      list[dict]
    }

For backend integration (Java Spring Boot):
    - Single answer : call evaluate_answer(question, candidate_answer, question_type)
    - Full session  : call evaluate_session(qa_pairs)
    - Save results  : call save_evaluation_result(result, output_path)

Dependencies:
    pip install groq python-dotenv
"""

import os
import json

# ── Import from shared/ instead of defining locally ──
from shared.groq_client import get_groq_client, MODEL_CONFIGS
from shared.retry_handler import call_with_retry, parse_json_response

# Import prompt builder functions
try:
    from .prompt_templates import (
        get_evaluation_prompt,
        get_batch_evaluation_prompt,
        get_batch_summary_prompt,
    )
except ImportError:
    from prompt_templates import (
        get_evaluation_prompt,
        get_batch_evaluation_prompt,
        get_batch_summary_prompt,
    )


# ─────────────────────────────────────────────
# SECTION: Scoring Weights per Criteria
# ─────────────────────────────────────────────

CRITERIA_WEIGHTS = {
    "relevance":          0.30,
    "technical_accuracy": 0.30,
    "clarity":            0.20,
    "completeness":       0.20,
}


# ─────────────────────────────────────────────
# SECTION: Performance Labels
# ─────────────────────────────────────────────

def get_performance_label(score: float) -> str:
    """
    Returns a human-readable performance label based on the score.

    Parameters:
        score (float): The overall evaluation score (0–100).

    Returns:
        str: One of: "Excellent", "Good", "Average", "Needs Improvement", "Poor"
    """
    if score >= 85:
        return "Excellent"
    elif score >= 70:
        return "Good"
    elif score >= 55:
        return "Average"
    elif score >= 40:
        return "Needs Improvement"
    else:
        return "Poor"


# ─────────────────────────────────────────────
# SECTION: LLM Call Helper
# ─────────────────────────────────────────────

def _call_for_evaluation(client, prompt: str) -> dict | None:
    """
    Sends an evaluation prompt to Groq via the shared retry handler
    and parses the JSON response.

    Parameters:
        client:        Shared Groq client from shared/groq_client.py
        prompt (str):  The fully formatted evaluation prompt.

    Returns:
        dict | None: Parsed evaluation dict, or None if parsing fails.
    """
    config = MODEL_CONFIGS["answer_evaluator"]

    raw_text = call_with_retry(
        client       = client,
        messages     = [
            {
                "role": "system",
                "content": (
                    "You are an expert interview coach. "
                    "Always respond with ONLY a valid JSON object. "
                    "No explanation, no markdown formatting, no extra text."
                )
            },
            {"role": "user", "content": prompt}
        ],
        model        = config["model"],
        temperature  = config["temperature"],
        max_tokens   = config["max_tokens"],
        caller_label = "AnswerEvaluator",
    )

    return parse_json_response(raw_text, "AnswerEvaluator")


# ─────────────────────────────────────────────
# SECTION: Single Answer Evaluator
# ─────────────────────────────────────────────

def evaluate_answer(question: str, candidate_answer: str, question_type: str = "technical") -> dict:
    """
    Evaluates a single candidate answer to an interview question.
    This is the PRIMARY function for evaluating one answer at a time.

    Parameters:
        question (str):
            The interview question that was asked.
            Example: "What is the difference between REST and GraphQL?"

        candidate_answer (str):
            The candidate's spoken or typed response.
            Example: "REST uses endpoints while GraphQL uses a single endpoint..."

        question_type (str):
            The type of question. Used to adjust evaluation context.
            Must be one of: "technical", "behavioral", "hr_general", "resume_based"
            Defaults to "technical"

    Returns:
        dict: A structured evaluation result.

        Example return value:
        {
            "question":         "What is the difference between REST and GraphQL?",
            "question_type":    "technical",
            "candidate_answer": "REST uses endpoints while GraphQL...",
            "overall_score":    77.5,
            "performance":      "Good",
            "breakdown": {
                "relevance":          {"score": 85, "feedback": "Directly addresses the question."},
                "technical_accuracy": {"score": 75, "feedback": "Correct but missed subscriptions."},
                "clarity":            {"score": 80, "feedback": "Clear and well structured."},
                "completeness":       {"score": 70, "feedback": "Did not mention use cases."}
            },
            "ideal_answer": "REST and GraphQL are both API design approaches..."
        }

        On failure, returns:
        {
            "error": "Reason for failure"
        }

    Example usage (Python):
        from answer_evaluator.answer_evaluator import evaluate_answer

        result = evaluate_answer(
            question         = "Explain what Docker is and why it is used.",
            candidate_answer = "Docker is a containerization tool...",
            question_type    = "technical"
        )
        print(result["overall_score"])
        print(result["ideal_answer"])
    """

    # ── Step 1: Validate inputs ──
    if not question or not question.strip():
        return {"error": "question is empty or None."}
    if not candidate_answer or not candidate_answer.strip():
        return {"error": "candidate_answer is empty or None."}

    valid_types = ["technical", "behavioral", "hr_general", "resume_based"]
    if question_type not in valid_types:
        print(f"[AnswerEvaluator] WARNING: Unknown question_type '{question_type}'. Defaulting to 'technical'.")
        question_type = "technical"

    # ── Step 2: Get shared Groq client ──
    try:
        client = get_groq_client()
    except ValueError as e:
        return {"error": str(e)}

    # ── Step 3: Build prompt and call Groq ──
    prompt     = get_evaluation_prompt(question, candidate_answer, question_type)
    raw_result = _call_for_evaluation(client, prompt)

    if raw_result is None:
        return {"error": "Failed to get evaluation from Groq API. Please retry."}

    # ── Step 4: Extract criteria scores and feedback ──
    breakdown = {}
    for criteria in ["relevance", "technical_accuracy", "clarity", "completeness"]:
        criteria_data = raw_result.get(criteria, {})
        breakdown[criteria] = {
            "score":    max(0, min(100, int(criteria_data.get("score", 0)))),
            "feedback": str(criteria_data.get("feedback", "No feedback provided.")),
        }

    # ── Step 5: Calculate weighted overall score ──
    overall_score = sum(
        breakdown[c]["score"] * CRITERIA_WEIGHTS[c]
        for c in CRITERIA_WEIGHTS
    )
    overall_score = round(overall_score, 1)

    # ── Step 6: Get ideal answer ──
    ideal_answer = str(raw_result.get("ideal_answer", "No ideal answer provided."))

    return {
        "question":         question,
        "question_type":    question_type,
        "candidate_answer": candidate_answer,
        "overall_score":    overall_score,
        "performance":      get_performance_label(overall_score),
        "breakdown":        breakdown,
        "ideal_answer":     ideal_answer,
    }


# ─────────────────────────────────────────────
# SECTION: Full Session Evaluator
# ─────────────────────────────────────────────

def evaluate_session(qa_pairs: list) -> dict:
    """
    Evaluates multiple answers from a complete mock interview session
    and generates an overall performance summary.

    Parameters:
        qa_pairs (list):
            A list of dicts, each containing a question and answer.
            Each dict must have:
                - "question"      (str) — the interview question
                - "answer"        (str) — the candidate's answer
                - "question_type" (str) — optional, defaults to "technical"

            Example:
            [
                {"question": "What is Docker?", "answer": "Docker is...", "question_type": "technical"},
                {"question": "Tell me about yourself.", "answer": "I am...", "question_type": "hr_general"}
            ]

    Returns:
        dict: A full session evaluation report.

        Example return value:
        {
            "total_questions":  2,
            "average_score":    74.5,
            "overall_feedback": "Strong technical skills but needs improvement in HR answers.",
            "strengths":        ["Technical knowledge", "Clear explanations"],
            "improvements":     ["Use STAR method", "Be more concise in HR answers"],
            "evaluations":      [ ...individual results... ]
        }

    Example usage (Python):
        from answer_evaluator.answer_evaluator import evaluate_session

        qa_pairs = [
            {"question": "What is Docker?", "answer": "Docker is...", "question_type": "technical"},
            {"question": "Tell me about yourself.", "answer": "I am...", "question_type": "hr_general"},
        ]
        report = evaluate_session(qa_pairs)
        print(report["average_score"])
        print(report["strengths"])
    """

    if not qa_pairs or not isinstance(qa_pairs, list):
        return {"error": "qa_pairs must be a non-empty list of question-answer dicts."}

    # ── Get shared Groq client ──
    try:
        client = get_groq_client()
    except ValueError as e:
        return {"error": str(e)}

    evaluations = []
    total_score = 0
    config      = MODEL_CONFIGS["answer_evaluator"]

    print(f"[AnswerEvaluator] Evaluating {len(qa_pairs)} answers in batches of 5...")

    # ── Process answers in batches of 5 (saves API calls) ──
    # e.g. 20 answers = 4 batches = 4 API calls instead of 20
    batch_size = 5
    for batch_start in range(0, len(qa_pairs), batch_size):
        batch     = qa_pairs[batch_start: batch_start + batch_size]
        batch_num = (batch_start // batch_size) + 1
        print(f"[AnswerEvaluator] Processing batch {batch_num} ({len(batch)} answers)...")

        prompt = get_batch_evaluation_prompt(batch)
        raw_response = call_with_retry(
            client       = client,
            messages     = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert interview coach. "
                        "Always respond with ONLY a valid JSON object. "
                        "No explanation, no markdown, no extra text."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            model        = config["model"],
            temperature  = config["temperature"],
            max_tokens   = 2048,   # Higher since evaluating up to 5 answers at once
            caller_label = f"AnswerEvaluator:batch{batch_num}",
        )

        batch_result = parse_json_response(raw_response, f"AnswerEvaluator:batch{batch_num}")
        batch_evals  = []

        if batch_result and isinstance(batch_result, dict):
            raw_evals = batch_result.get("evaluations", [])
            for i, raw in enumerate(raw_evals):
                pair          = batch[i] if i < len(batch) else {}
                question      = pair.get("question", "")
                answer        = pair.get("answer", "")
                question_type = pair.get("question_type", "technical")

                breakdown = {}
                for criteria in ["relevance", "technical_accuracy", "clarity", "completeness"]:
                    criteria_data = raw.get(criteria, {})
                    breakdown[criteria] = {
                        "score":    max(0, min(100, int(criteria_data.get("score", 0)))),
                        "feedback": str(criteria_data.get("feedback", "No feedback provided.")),
                    }

                overall_score = round(sum(
                    breakdown[c]["score"] * CRITERIA_WEIGHTS[c]
                    for c in CRITERIA_WEIGHTS
                ), 1)

                eval_result = {
                    "question":         question,
                    "question_type":    question_type,
                    "candidate_answer": answer,
                    "overall_score":    overall_score,
                    "performance":      get_performance_label(overall_score),
                    "breakdown":        breakdown,
                    "ideal_answer":     str(raw.get("ideal_answer", "No ideal answer provided.")),
                }
                batch_evals.append(eval_result)
                total_score += overall_score
        else:
            # Fallback: if batch failed, evaluate individually
            print(f"[AnswerEvaluator] Batch {batch_num} failed, falling back to individual evaluation...")
            for pair in batch:
                result = evaluate_answer(
                    pair.get("question", ""),
                    pair.get("answer", ""),
                    pair.get("question_type", "technical")
                )
                if "error" not in result:
                    total_score += result.get("overall_score", 0)
                batch_evals.append(result)

        evaluations.extend(batch_evals)

    # ── Calculate average score ──
    valid_evals   = [e for e in evaluations if "error" not in e]
    average_score = round(total_score / len(valid_evals), 1) if valid_evals else 0

    # ── Generate overall session summary via Groq ──
    summary = {"overall_feedback": "", "strengths": [], "improvements": []}

    if valid_evals:
        try:
            client         = get_groq_client()
            summary_prompt = get_batch_summary_prompt(valid_evals)
            raw_summary    = _call_for_evaluation(client, summary_prompt)
            if raw_summary:
                summary["overall_feedback"] = str(raw_summary.get("overall_feedback", ""))
                summary["strengths"]        = list(raw_summary.get("strengths", []))
                summary["improvements"]     = list(raw_summary.get("improvements", []))
        except Exception as e:
            print(f"[AnswerEvaluator] WARNING: Could not generate session summary: {e}")

    print(f"[AnswerEvaluator] Session complete! Average score: {average_score}")

    return {
        "total_questions":  len(qa_pairs),
        "average_score":    average_score,
        "overall_feedback": summary["overall_feedback"],
        "strengths":        summary["strengths"],
        "improvements":     summary["improvements"],
        "evaluations":      evaluations,
    }


# ─────────────────────────────────────────────
# SECTION: Save Output to JSON
# ─────────────────────────────────────────────

def save_evaluation_result(result: dict, output_path: str = "output/evaluation_result.json") -> None:
    """
    Saves the evaluation result dictionary to a JSON file.

    Parameters:
        result (dict):      Result from evaluate_answer() or evaluate_session().
        output_path (str):  Path to save the JSON file.
                            Defaults to "output/evaluation_result.json"

    Returns:
        None
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f"[AnswerEvaluator] Result saved to: {output_path}")


# ─────────────────────────────────────────────
# SECTION: Quick Test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    OUTPUT_PATH = "output/evaluation_result.json"

    sample_qa_pairs = [
        {
            "question": "What is the difference between REST and GraphQL?",
            "answer": (
                "REST uses multiple endpoints for different resources, "
                "while GraphQL uses a single endpoint where the client "
                "specifies exactly what data it needs. GraphQL reduces "
                "over-fetching and under-fetching of data."
            ),
            "question_type": "technical"
        },
        {
            "question": "Tell me about a time you worked under pressure to meet a deadline.",
            "answer": (
                "During my internship, we had a product demo in 2 days and "
                "a critical feature was broken. I stayed late, debugged the issue, "
                "communicated progress to my team lead, and we delivered on time."
            ),
            "question_type": "behavioral"
        },
        {
            "question": "Why are you interested in this role?",
            "answer": (
                "I am passionate about backend development and this role aligns "
                "perfectly with my skills in Java and Spring Boot. I want to grow "
                "in a company that values engineering excellence."
            ),
            "question_type": "hr_general"
        },
    ]

    print("[AnswerEvaluator] Starting test session evaluation...\n")
    report = evaluate_session(sample_qa_pairs)

    if "error" in report:
        print(f"[AnswerEvaluator] FAILED: {report['error']}")
    else:
        print(f"\n[AnswerEvaluator] SESSION REPORT")
        print(f"  Total Questions : {report['total_questions']}")
        print(f"  Average Score   : {report['average_score']} / 100")
        print(f"  Overall Feedback: {report['overall_feedback']}")
        print(f"  Strengths       : {report['strengths']}")
        print(f"  Improvements    : {report['improvements']}")
        print(f"\n  Individual Evaluations:")
        for i, ev in enumerate(report["evaluations"], 1):
            if "error" in ev:
                print(f"\n  Q{i}: ERROR — {ev['error']}")
            else:
                print(f"\n  Q{i} [{ev['question_type'].upper()}]: {ev['question']}")
                print(f"    Overall Score : {ev['overall_score']} / 100 ({ev['performance']})")
                for criteria, data in ev["breakdown"].items():
                    print(f"    {criteria:22}: {data['score']}/100 — {data['feedback']}")
                print(f"    Ideal Answer  : {ev['ideal_answer'][:120]}...")

        save_evaluation_result(report, OUTPUT_PATH)