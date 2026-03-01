from django.contrib import admin
from .models import Problem, Submission


@admin.register(Problem)
class ProblemAdmin(admin.ModelAdmin):
    list_display = ["title", "category", "subject_area", "submission_count", "created_at"]
    list_filter = ["category", "subject_area"]
    search_fields = ["title", "description"]
    readonly_fields = ["created_at", "updated_at"]

    def submission_count(self, obj):
        return obj.submissions.count()
    submission_count.short_description = "Submissions"


@admin.register(Submission)
class SubmissionAdmin(admin.ModelAdmin):
    list_display = ["pk", "problem", "status", "score", "submitted_at"]
    list_filter = ["status", "problem__category", "problem"]
    readonly_fields = ["submitted_at", "graded_at"]
