from django.urls import path
from . import views

urlpatterns = [
    # Home
    path("", views.home, name="home"),
    # Category lists
    path("coding/", views.coding_list, name="coding_list"),
    path("coding/<str:subject>/", views.coding_list, name="coding_list_subject"),
    path("theory/", views.theory_list, name="theory_list"),
    path("theory/<str:subject>/", views.theory_list, name="theory_list_subject"),
    # Problems CRUD
    path("problems/new/", views.problem_create, name="problem_create"),
    path("problems/<int:pk>/", views.problem_detail, name="problem_detail"),
    path("problems/<int:pk>/edit/", views.problem_edit, name="problem_edit"),
    path("problems/<int:pk>/delete/", views.problem_delete, name="problem_delete"),
    # Submissions
    path("problems/<int:problem_pk>/submit/", views.submission_create, name="submission_create"),
    path("submissions/<int:pk>/", views.submission_detail, name="submission_detail"),
]
