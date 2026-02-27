from django.db import models


class Problem(models.Model):
    title = models.CharField(max_length=200)
    description = models.TextField(help_text="Full problem description shown to students.")
    grading_criteria = models.TextField(
        help_text="Criteria Claude will use to evaluate submissions (e.g. correctness, edge cases, style)."
    )
    example_input = models.TextField(blank=True, help_text="Optional example input for the student.")
    example_output = models.TextField(blank=True, help_text="Optional expected output for the student.")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.title


class Submission(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        GRADING = "grading", "Grading"
        GRADED = "graded", "Graded"
        ERROR = "error", "Error"

    problem = models.ForeignKey(Problem, on_delete=models.CASCADE, related_name="submissions")
    code = models.TextField()
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
