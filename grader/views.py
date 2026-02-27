from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from django.utils import timezone

from .models import Problem, Submission
from .forms import ProblemForm, SubmissionForm
from .grading import grade_submission


# ── Problems ──────────────────────────────────────────────────────────────────

def problem_list(request):
    problems = Problem.objects.all()
    return render(request, "grader/problem_list.html", {"problems": problems})


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


def problem_detail(request, pk):
    problem = get_object_or_404(Problem, pk=pk)
    submissions = problem.submissions.all()
    form = SubmissionForm()
    return render(request, "grader/problem_detail.html", {
        "problem": problem,
        "submissions": submissions,
        "form": form,
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
    return render(request, "grader/problem_form.html", {"form": form, "action": "Edit", "problem": problem})


def problem_delete(request, pk):
    problem = get_object_or_404(Problem, pk=pk)
    if request.method == "POST":
        title = problem.title
        problem.delete()
        messages.success(request, f"Problem '{title}' deleted.")
        return redirect("problem_list")
    return render(request, "grader/problem_confirm_delete.html", {"problem": problem})


# ── Submissions ───────────────────────────────────────────────────────────────

def submission_create(request, problem_pk):
    problem = get_object_or_404(Problem, pk=problem_pk)
    if request.method == "POST":
        form = SubmissionForm(request.POST)
        if form.is_valid():
            submission = form.save(commit=False)
            submission.problem = problem
            submission.status = Submission.Status.GRADING
            submission.save()

            try:
                result = grade_submission(problem, submission.code)
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
        form = SubmissionForm()

    return render(request, "grader/problem_detail.html", {
        "problem": problem,
        "submissions": problem.submissions.all(),
        "form": form,
    })


def submission_detail(request, pk):
    submission = get_object_or_404(Submission, pk=pk)
    return render(request, "grader/submission_detail.html", {"submission": submission})
