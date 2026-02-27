"""
Grading service — calls the Anthropic API to evaluate a code submission.
"""
import json
import os

import anthropic


GRADING_SYSTEM_PROMPT = """You are an expert Python programming instructor and code grader.
You will be given a programming problem, the grading criteria, and a student's code submission.
Your job is to evaluate the submission fairly and thoroughly.

You MUST respond with ONLY a valid JSON object in this exact format:
{
    "score": <integer 0-100>,
    "summary": "<one sentence overall assessment>",
    "feedback": "<detailed markdown feedback with sections for: Correctness, Code Quality, Edge Cases, and Suggestions for Improvement>"
}

Do not include any text outside the JSON object."""


def grade_submission(problem, code: str) -> dict:
    """
    Sends the problem + code to Claude and returns a dict with:
        score (int), summary (str), feedback (str)
    Raises an exception on API or parse errors.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Please set it before running the server."
        )

    client = anthropic.Anthropic(api_key=api_key)

    example_section = ""
    if problem.example_input or problem.example_output:
        example_section = f"""
**Example Input:**
```
{problem.example_input}
```
**Example Output:**
```
{problem.example_output}
```"""

    user_message = f"""## Problem: {problem.title}

### Description
{problem.description}
{example_section}

### Grading Criteria
{problem.grading_criteria}

### Student's Submission
```python
{code}
```

Please evaluate this submission and respond with the JSON grading result."""

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=GRADING_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    raw = message.content[0].text.strip()

    # Strip markdown code fences if Claude wrapped it
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

    result = json.loads(raw)

    score = int(result.get("score", 0))
    score = max(0, min(100, score))

    return {
        "score": score,
        "summary": result.get("summary", ""),
        "feedback": result.get("feedback", ""),
    }
