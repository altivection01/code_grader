from django import forms
from .models import Problem, Submission, ALL_SUBJECT_AREAS, CODING_SUBJECT_AREAS, THEORY_SUBJECT_AREAS


class ProblemForm(forms.ModelForm):
    class Meta:
        model = Problem
        fields = ["category", "subject_area", "title", "description", "grading_criteria", "example_input", "example_output"]
        widgets = {
            "category": forms.Select(attrs={"class": "form-select"}),
            "subject_area": forms.Select(attrs={"class": "form-select"}),
            "title": forms.TextInput(attrs={"class": "form-control", "placeholder": "e.g. FizzBuzz"}),
            "description": forms.Textarea(
                attrs={"class": "form-control", "rows": 6, "placeholder": "Describe the problem..."}
            ),
            "grading_criteria": forms.Textarea(
                attrs={"class": "form-control", "rows": 5, "placeholder": "List the criteria Claude will grade against..."}
            ),
            "example_input": forms.Textarea(
                attrs={"class": "form-control", "rows": 3, "placeholder": "Optional example input"}
            ),
            "example_output": forms.Textarea(
                attrs={"class": "form-control", "rows": 3, "placeholder": "Optional expected output"}
            ),
        }


class CodingSubmissionForm(forms.ModelForm):
    class Meta:
        model = Submission
        fields = ["answer"]
        widgets = {
            "answer": forms.Textarea(
                attrs={
                    "class": "form-control font-monospace",
                    "rows": 20,
                    "placeholder": "# Write your Python solution here...",
                    "spellcheck": "false",
                }
            )
        }
        labels = {"answer": "Your Code"}


class TheorySubmissionForm(forms.ModelForm):
    class Meta:
        model = Submission
        fields = ["answer"]
        widgets = {
            "answer": forms.Textarea(
                attrs={
                    "class": "form-control",
                    "rows": 18,
                    "placeholder": (
                        "Write your answer here. Markdown and LaTeX are both supported:\n\n"
                        "  Inline math:   $E = mc^2$\n"
                        "  Display math:  $$\\hat{\\beta} = (X^TX)^{-1}X^Ty$$\n\n"
                        "Structure your answer with headings, bullet points, and equations."
                    ),
                    "spellcheck": "true",
                    "style": "font-size: 0.95rem; line-height: 1.6;",
                }
            )
        }
        labels = {"answer": "Your Answer"}
