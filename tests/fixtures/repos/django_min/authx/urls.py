"""Authentication URL routes."""
from django.urls import path

from .views import LoginView, ProfileView, TokenRevokeView

app_name = "authx"

urlpatterns = [
    path("login", LoginView.as_view(), name="login"),
    path("profile", ProfileView.as_view(), name="profile"),
    path("tokens/<str:key>", TokenRevokeView.as_view(), name="token-revoke"),
]
