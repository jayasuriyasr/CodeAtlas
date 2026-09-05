"""Reusable model mixins."""
from django.db import models


class TimestampMixin(models.Model):
    """Adds created/updated stamps."""

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True

    def touch(self) -> None:
        self.save(update_fields=["updated_at"])


class SoftDeleteMixin(models.Model):
    deleted_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        abstract = True

    def soft_delete(self) -> None:
        from django.utils import timezone

        self.deleted_at = timezone.now()
        self.save(update_fields=["deleted_at"])

    def is_deleted(self) -> bool:
        return self.deleted_at is not None
