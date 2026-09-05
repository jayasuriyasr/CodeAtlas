"""Billing URL routes."""
from django.urls import path

from .views import SubscriptionView

app_name = "billing"

urlpatterns = [path("subscription", SubscriptionView.as_view(), name="subscription")]
