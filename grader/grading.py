"""
Grading service — calls the Anthropic API to evaluate submissions.
Dispatches to the appropriate grader based on problem category.
"""
import json
import os

import anthropic

MODEL = "claude-opus-4-6"

# ── Coding grader ─────────────────────────────────────────────────────────────

CODING_SYSTEM_PROMPT = """You are an expert Python programming instructor and code grader.
You will be given a programming problem, the grading criteria, and a student's code submission.
Your job is to evaluate the submission fairly and thoroughly.

You MUST respond with ONLY a valid JSON object in this exact format:
{
    "score": <integer 0-100>,
    "summary": "<one sentence overall assessment>",
    "feedback": "<detailed markdown feedback with sections for: Correctness, Code Quality, Edge Cases, and Suggestions for Improvement>"
}

Do not include any text outside the JSON object."""


def _grade_coding(client, problem, answer: str) -> dict:
    example_section = ""
    if problem.example_input or problem.example_output:
        example_section = (
            f"\n**Example Input:**\n```\n{problem.example_input}\n```\n"
            f"**Example Output:**\n```\n{problem.example_output}\n```"
        )
    user_message = f"""## Problem: {problem.title}

### Description
{problem.description}
{example_section}

### Grading Criteria
{problem.grading_criteria}

### Student's Submission
```python
{answer}
```

Please evaluate this submission and respond with the JSON grading result."""
    return _call_claude(client, CODING_SYSTEM_PROMPT, user_message)


# ── Theory grader ─────────────────────────────────────────────────────────────

THEORY_SYSTEM_PROMPT = """You are an expert ML/AI educator, mathematician, and examiner.
You will be given a theoretical ML/AI question, the grading criteria, and a student's written answer.
The student's answer may contain Markdown formatting and LaTeX mathematical notation.
Interpret all LaTeX correctly when evaluating.

Evaluate the answer rigorously across these dimensions:
1. Conceptual correctness - are the core ideas accurate?
2. Mathematical rigour - are formulas, derivations, and notation correct?
3. Completeness - does the answer address all parts of the question and criteria?
4. Clarity - is the explanation well-structured and understandable?

You MUST respond with ONLY a valid JSON object in this exact format:
{
    "score": <integer 0-100>,
    "summary": "<one sentence overall assessment>",
    "feedback": "<detailed markdown feedback with H2 sections: ## Conceptual Accuracy, ## Mathematical Correctness, ## Completeness, ## Clarity & Structure, ## Suggestions for Improvement. Use LaTeX where helpful in your feedback.>"
}

Scoring guide:
  90-100: All concepts correct, mathematically rigorous, complete, clearly explained
  75-89:  Mostly correct with minor errors or omissions
  60-74:  Core ideas present but notable gaps or mathematical errors
  40-59:  Partial understanding, significant gaps
  0-39:   Fundamental misunderstandings or very incomplete

Do not include any text outside the JSON object."""


def _grade_theory(client, problem, answer: str) -> dict:
    user_message = f"""## Question: {problem.title}

### Full Question
{problem.description}

### Grading Criteria
{problem.grading_criteria}

### Student's Answer
{answer}

Please evaluate this answer and respond with the JSON grading result."""
    return _call_claude(client, THEORY_SYSTEM_PROMPT, user_message)


# ── Shared helper ─────────────────────────────────────────────────────────────

def _call_claude(client, system_prompt: str, user_message: str) -> dict:
    message = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
    result = json.loads(raw)
    score = max(0, min(100, int(result.get("score", 0))))
    return {
        "score": score,
        "summary": result.get("summary", ""),
        "feedback": result.get("feedback", ""),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def grade_submission(problem, answer: str) -> dict:
    """
    Grade a submission against its problem.  Dispatches to the correct
    grader based on problem.category.  Returns dict: score, summary, feedback.
    Raises an exception on API or parse errors.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Please set it before running the server."
        )
    client = anthropic.Anthropic(api_key=api_key)
    if problem.is_theory:
        return _grade_theory(client, problem, answer)
    return _grade_coding(client, problem, answer)
