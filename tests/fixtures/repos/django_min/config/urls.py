"""Root URL configuration."""
from django.urls import include, path

from authx.views import HealthView

urlpatterns = [
    path("api/auth/", include("authx.urls")),
    path("api/billing/", include("billing.urls")),
    path("healthz", HealthView.as_view(), name="health"),
]
