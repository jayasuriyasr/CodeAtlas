"""Authentication models."""
from django.contrib.auth.models import AbstractUser
from django.db import models

from common.mixins import TimestampMixin


class User(TimestampMixin, AbstractUser):
    """A platform user."""

    email = models.EmailField(unique=True)
    is_verified = models.BooleanField(default=False)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.email

    def display_name(self) -> str:
        """Prefer the full name, fall back to the email local part."""
        full = self.get_full_name()
        return full or self.email.split("@")[0]

    def mark_verified(self, *, save: bool = True) -> None:
        self.is_verified = True
        if save:
            self.save(update_fields=["is_verified"])


class ApiToken(TimestampMixin, models.Model):
    """An opaque bearer token."""

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="tokens")
    key = models.CharField(max_length=64, unique=True)
    revoked_at = models.DateTimeField(null=True, blank=True)

    def is_active(self) -> bool:
        return self.revoked_at is None

    def revoke(self) -> None:
        from django.utils import timezone

        self.revoked_at = timezone.now()
        self.save(update_fields=["revoked_at"])
