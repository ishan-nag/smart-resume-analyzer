"""
answer_evaluator/__init__.py
=============================
Exposes the public API of the answer_evaluator module.

Backend developers only need to import from here:

    from answer_evaluator import evaluate_answer, evaluate_session, save_evaluation_result

Available functions:
    - evaluate_answer(question, candidate_answer, question_type)  → dict
    - evaluate_session(qa_pairs)                                  → dict
    - save_evaluation_result(result, output_path)                 → None

─────────────────────────────────────────────────────────────────
Single Answer Input:
    question        (str) — the interview question
    candidate_answer(str) — the candidate's response
    question_type   (str) — "technical" | "behavioral" | "hr_general" | "resume_based"

Single Answer Output:
    {
        "question":         str,
        "question_type":    str,
        "candidate_answer": str,
        "overall_score":    float (0-100),
        "performance":      str ("Excellent" / "Good" / "Average" /
                                 "Needs Improvement" / "Poor"),
        "breakdown": {
            "relevance":          {"score": int, "feedback": str},
            "technical_accuracy": {"score": int, "feedback": str},
            "clarity":            {"score": int, "feedback": str},
            "completeness":       {"score": int, "feedback": str}
        },
        "ideal_answer": str
    }

─────────────────────────────────────────────────────────────────
Session Input:
    qa_pairs (list of dicts):
        [{"question": str, "answer": str, "question_type": str}, ...]

Session Output:
    {
        "total_questions":  int,
        "average_score":    float,
        "overall_feedback": str,
        "strengths":        list[str],
        "improvements":     list[str],
        "evaluations":      list[dict]
    }
"""

from answer_evaluator.answer_evaluator import (
    evaluate_answer,
    evaluate_session,
    save_evaluation_result,
)

__all__ = ["evaluate_answer", "evaluate_session", "save_evaluation_result"]