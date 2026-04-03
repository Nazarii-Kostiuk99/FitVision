from django.urls import path
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

from .views import (
    AnalyseView,
    AnalysisDetailView,
    AnalysisListView,
    RegisterView,
    MeView,
)

urlpatterns = [
    # authentication
    path("auth/register/", RegisterView.as_view()),
    path("auth/login/", TokenObtainPairView.as_view()),
    path("auth/refresh/", TokenRefreshView.as_view()),
    path("auth/me/", MeView.as_view()),
    # exercise analysis
    # /api
    path("analyse/", AnalyseView.as_view()),
    path("analysis/<int:pk>/", AnalysisDetailView.as_view()),
    path("analyses/", AnalysisListView.as_view()),
]
