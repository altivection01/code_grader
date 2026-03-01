from django.db import models

CODING_SUBJECT_AREAS = [
    ("python_basics", "Python Basics"),
    ("numpy", "NumPy"),
    ("pandas", "Pandas"),
    ("matplotlib", "Matplotlib"),
    ("seaborn", "Seaborn"),
    ("sklearn", "Scikit-learn"),
    ("tensorflow", "TensorFlow"),
]

THEORY_SUBJECT_AREAS = [
    ("regression", "Regression"),
    ("trees_forests", "Trees & Forests"),
    ("classification", "Classification & Clustering"),
    ("neural_networks", "Neural Networks"),
    ("generative_ai", "Generative AI for NLP"),
]

ALL_SUBJECT_AREAS = CODING_SUBJECT_AREAS + THEORY_SUBJECT_AREAS


class Problem(models.Model):
    class Category(models.TextChoices):
        CODING = "coding", "Python Coding"
        THEORY = "theory", "ML/AI Theory"

    title = models.CharField(max_length=200)
    description = models.TextField(help_text="Full problem description shown to students.")
    grading_criteria = models.TextField(
        help_text="Criteria Claude will use to evaluate submissions."
    )
    example_input = models.TextField(blank=True)
    example_output = models.TextField(blank=True)
    category = models.CharField(
        max_length=20,
        choices=Category.choices,
        default=Category.CODING,
        db_index=True,
    )
    subject_area = models.CharField(
        max_length=50,
        choices=ALL_SUBJECT_AREAS,
        default="python_basics",
        db_index=True,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["subject_area", "title"]

    def __str__(self):
        return self.title

    @property
    def subject_label(self):
        return dict(ALL_SUBJECT_AREAS).get(self.subject_area, self.subject_area)

    @property
    def is_theory(self):
        return self.category == self.Category.THEORY

    @property
    def is_coding(self):
        return self.category == self.Category.CODING


class Submission(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        GRADING = "grading", "Grading"
        GRADED = "graded", "Graded"
        ERROR = "error", "Error"

    problem = models.ForeignKey(Problem, on_delete=models.CASCADE, related_name="submissions")
    # Stores Python code for coding problems; written prose/LaTeX for theory problems
    answer = models.TextField()
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING)
    score = models.IntegerField(null=True, blank=True, help_text="Score out of 100")
    feedback = models.TextField(blank=True, help_text="Claude's detailed feedback")
    error_message = models.TextField(blank=True)
    submitted_at = models.DateTimeField(auto_now_add=True)
    graded_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-submitted_at"]

    def __str__(self):
        return f"Submission #{self.pk} for '{self.problem.title}' — {self.status}"

    @property
    def score_letter(self):
        if self.score is None:
            return "N/A"
        if self.score >= 90:
            return "A"
        if self.score >= 80:
            return "B"
        if self.score >= 70:
            return "C"
        if self.score >= 60:
            return "D"
        return "F"

    @property
    def score_color(self):
        if self.score is None:
            return "secondary"
        if self.score >= 90:
            return "success"
        if self.score >= 70:
            return "warning"
        return "danger"
