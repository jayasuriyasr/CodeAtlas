"""Billing models."""
from django.db import models

from authx.models import User
from common.mixins import SoftDeleteMixin, TimestampMixin


class Plan(TimestampMixin, models.Model):
    """A subscription tier."""

    code = models.CharField(max_length=32, unique=True)
    monthly_cents = models.IntegerField(default=0)

    def price_display(self) -> str:
        return f"${self.monthly_cents / 100:.2f}"


class Subscription(TimestampMixin, SoftDeleteMixin, models.Model):
    """Links a user to a plan."""

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    plan = models.ForeignKey(Plan, on_delete=models.PROTECT)
    active = models.BooleanField(default=True)

    def monthly_total(self) -> int:
        return self.plan.monthly_cents if self.active else 0

    def cancel(self) -> None:
        self.active = False
        self.soft_delete()
