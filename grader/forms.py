from django import forms
from .models import Problem, Submission


class ProblemForm(forms.ModelForm):
    class Meta:
        model = Problem
        fields = ["title", "description", "grading_criteria", "example_input", "example_output"]
        widgets = {
            "title": forms.TextInput(attrs={"class": "form-control", "placeholder": "e.g. FizzBuzz"}),
            "description": forms.Textarea(
                attrs={"class": "form-control", "rows": 6, "placeholder": "Describe the problem..."}
            ),
            "grading_criteria": forms.Textarea(
                attrs={
                    "class": "form-control",
                    "rows": 5,
                    "placeholder": (
                        "e.g.\n"
                        "- Correctly handles numbers 1-100\n"
                        "- Prints 'Fizz' for multiples of 3\n"
                        "- Prints 'Buzz' for multiples of 5\n"
                        "- Uses a function\n"
                        "- Has clear variable names"
                    ),
                }
            ),
            "example_input": forms.Textarea(
                attrs={"class": "form-control", "rows": 3, "placeholder": "Optional example input"}
            ),
            "example_output": forms.Textarea(
                attrs={"class": "form-control", "rows": 3, "placeholder": "Optional expected output"}
            ),
        }


class SubmissionForm(forms.ModelForm):
    class Meta:
        model = Submission
        fields = ["code"]
        widgets = {
            "code": forms.Textarea(
                attrs={
                    "class": "form-control font-monospace",
                    "rows": 20,
                    "placeholder": "# Write your Python solution here...",
                    "spellcheck": "false",
                }
            )
        }
