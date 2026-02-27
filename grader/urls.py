from django.urls import path
from . import views

urlpatterns = [
    # Problems
    path("", views.problem_list, name="problem_list"),
    path("problems/new/", views.problem_create, name="problem_create"),
    path("problems/<int:pk>/", views.problem_detail, name="problem_detail"),
    path("problems/<int:pk>/edit/", views.problem_edit, name="problem_edit"),
    path("problems/<int:pk>/delete/", views.problem_delete, name="problem_delete"),
    # Submissions
    path("problems/<int:problem_pk>/submit/", views.submission_create, name="submission_create"),
    path("submissions/<int:pk>/", views.submission_detail, name="submission_detail"),
]
