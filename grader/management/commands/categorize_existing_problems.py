"""
Sets category='coding' and infers subject_area from the title prefix
for all problems that still have the default subject_area='python_basics'.
Safe to run multiple times (idempotent).
"""
from django.core.management.base import BaseCommand
from grader.models import Problem

TITLE_PREFIX_MAP = {
    "[Python Basics]":  "python_basics",
    "[NumPy]":          "numpy",
    "[Pandas]":         "pandas",
    "[Matplotlib]":     "matplotlib",
    "[Seaborn]":        "seaborn",
    "[Scikit-learn]":   "sklearn",
    "[TensorFlow]":     "tensorflow",
}

class Command(BaseCommand):
    help = "Infer and set subject_area for existing coding problems from their title prefix."

    def handle(self, *args, **options):
        updated = 0
        for problem in Problem.objects.filter(category="coding"):
            for prefix, subject in TITLE_PREFIX_MAP.items():
                if problem.title.startswith(prefix):
                    if problem.subject_area != subject:
                        problem.subject_area = subject
                        problem.save(update_fields=["subject_area"])
                        updated += 1
                        self.stdout.write(f"  ✓ {problem.title[:60]} → {subject}")
                    break
        self.stdout.write(self.style.SUCCESS(f"\nUpdated {updated} problem(s)."))
