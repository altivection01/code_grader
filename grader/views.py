from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from django.utils import timezone
from django.db.models import Count

from .models import (
    Problem, Submission,
    CODING_SUBJECT_AREAS, THEORY_SUBJECT_AREAS,
)
from .forms import ProblemForm, CodingSubmissionForm, TheorySubmissionForm
from .grading import grade_submission


# ── Home ──────────────────────────────────────────────────────────────────────

def home(request):
    coding_counts = {
        s: Problem.objects.filter(category="coding", subject_area=s).count()
        for s, _ in CODING_SUBJECT_AREAS
    }
    theory_counts = {
        s: Problem.objects.filter(category="theory", subject_area=s).count()
        for s, _ in THEORY_SUBJECT_AREAS
    }
    return render(request, "home.html", {
        "coding_subjects": [(s, label, coding_counts[s]) for s, label in CODING_SUBJECT_AREAS],
        "theory_subjects": [(s, label, theory_counts[s]) for s, label in THEORY_SUBJECT_AREAS],
        "total_coding": sum(coding_counts.values()),
        "total_theory": sum(theory_counts.values()),
    })


# ── Problem lists ─────────────────────────────────────────────────────────────

def coding_list(request, subject=None):
    qs = Problem.objects.filter(category="coding")
    active_subject = subject
    if subject:
        qs = qs.filter(subject_area=subject)
    return render(request, "grader/problem_list.html", {
        "problems": qs,
        "category": "coding",
        "category_label": "Python Coding",
        "subjects": CODING_SUBJECT_AREAS,
        "active_subject": active_subject,
        "accent": "primary",
        "icon": "bi-code-slash",
        "back_url": "coding_list",
    })


def theory_list(request, subject=None):
    qs = Problem.objects.filter(category="theory")
    active_subject = subject
    if subject:
        qs = qs.filter(subject_area=subject)
    return render(request, "grader/problem_list.html", {
        "problems": qs,
        "category": "theory",
        "category_label": "ML/AI Theory",
        "subjects": THEORY_SUBJECT_AREAS,
        "active_subject": active_subject,
        "accent": "success",
        "icon": "bi-journal-text",
        "back_url": "theory_list",
    })


# ── Problems CRUD ─────────────────────────────────────────────────────────────

def problem_create(request):
    if request.method == "POST":
        form = ProblemForm(request.POST)
        if form.is_valid():
            problem = form.save()
            messages.success(request, f"Problem '{problem.title}' created successfully.")
            return redirect("problem_detail", pk=problem.pk)
    else:
        form = ProblemForm()
    return render(request, "grader/problem_form.html", {"form": form, "action": "Create"})


def _problem_ui(problem):
    """Return accent/icon/submit_label context vars based on problem category."""
    if problem.is_theory:
        return {"accent": "success", "icon": "bi-journal-text", "submit_label": "Submit Answer"}
    return {"accent": "primary", "icon": "bi-code-slash", "submit_label": "Submit & Grade with Claude"}


def problem_detail(request, pk):
    problem = get_object_or_404(Problem, pk=pk)
    submissions = problem.submissions.all()
    form = TheorySubmissionForm() if problem.is_theory else CodingSubmissionForm()
    return render(request, "grader/problem_detail.html", {
        "problem": problem,
        "submissions": submissions,
        "form": form,
        **_problem_ui(problem),
    })


def problem_edit(request, pk):
    problem = get_object_or_404(Problem, pk=pk)
    if request.method == "POST":
        form = ProblemForm(request.POST, instance=problem)
        if form.is_valid():
            form.save()
            messages.success(request, "Problem updated.")
            return redirect("problem_detail", pk=problem.pk)
    else:
        form = ProblemForm(instance=problem)
    return render(request, "grader/problem_form.html", {
        "form": form, "action": "Edit", "problem": problem,
    })


def problem_delete(request, pk):
    problem = get_object_or_404(Problem, pk=pk)
    if request.method == "POST":
        title = problem.title
        cat = problem.category
        problem.delete()
        messages.success(request, f"Problem '{title}' deleted.")
        return redirect("coding_list" if cat == "coding" else "theory_list")
    return render(request, "grader/problem_confirm_delete.html", {"problem": problem})


# ── Submissions ───────────────────────────────────────────────────────────────

def submission_create(request, problem_pk):
    problem = get_object_or_404(Problem, pk=problem_pk)
    FormClass = TheorySubmissionForm if problem.is_theory else CodingSubmissionForm
    if request.method == "POST":
        form = FormClass(request.POST)
        if form.is_valid():
            submission = form.save(commit=False)
            submission.problem = problem
            submission.status = Submission.Status.GRADING
            submission.save()
            try:
                result = grade_submission(problem, submission.answer)
                submission.score = result["score"]
                submission.feedback = result["feedback"]
                submission.status = Submission.Status.GRADED
                submission.graded_at = timezone.now()
            except Exception as exc:
                submission.status = Submission.Status.ERROR
                submission.error_message = str(exc)
            submission.save()
            return redirect("submission_detail", pk=submission.pk)
    else:
        form = FormClass()

    return render(request, "grader/problem_detail.html", {
        "problem": problem,
        "submissions": problem.submissions.all(),
        "form": form,
        **_problem_ui(problem),
    })


def submission_detail(request, pk):
    submission = get_object_or_404(Submission, pk=pk)
    accent = "success" if submission.problem.is_theory else "primary"
    return render(request, "grader/submission_detail.html", {"submission": submission, "accent": accent})
