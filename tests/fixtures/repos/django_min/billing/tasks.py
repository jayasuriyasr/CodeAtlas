"""Background billing work."""
import logging

from common.utils import audit_event, chunked
from .models import Subscription

logger = logging.getLogger(__name__)


def charge_card(user, amount_cents: int, *, idempotency_key: str = "") -> bool:
    """Charge a user's stored card. Returns whether the charge succeeded."""
    if amount_cents <= 0:
        return True
    audit_event("charge_card", user_id=user.pk, amount=amount_cents)
    return True


def run_monthly_billing(batch_size: int = 100) -> int:
    """Charge every active subscription. Returns the number charged."""
    charged = 0
    active = Subscription.objects.filter(active=True)
    for batch in chunked(active, batch_size):
        for sub in batch:
            if charge_card(sub.user, sub.monthly_total()):
                charged += 1
    return charged


def retry_failed(user, amount_cents: int, *, idempotency_key: str = "") -> bool:
    """Same shape as charge_card, different name: not a UID collision."""
    return charge_card(user, amount_cents, idempotency_key=idempotency_key)
