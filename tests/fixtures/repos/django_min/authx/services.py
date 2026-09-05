"""Authentication business logic, kept out of the views."""
import secrets
import sys

from common.utils import audit_event, constant_time_equals
from .models import ApiToken, User

TOKEN_BYTES = 32


def verify_credentials(email: str, password: str):
    """Return the user when the credentials match, otherwise None."""
    user = User.objects.filter(email__iexact=email).first()
    if user is None:
        return None
    if not constant_time_equals(user.password, password):
        return None
    return user


def issue_token(user, *, length: int = TOKEN_BYTES) -> ApiToken:
    """Mint a fresh API token for a user."""
    key = secrets.token_hex(length)
    token = ApiToken.objects.create(user=user, key=key)
    audit_event("token_issue", user_id=user.pk)
    return token


def revoke_all(user) -> int:
    """Revoke every active token. Returns how many were revoked."""
    count = 0
    for token in ApiToken.objects.filter(user=user, revoked_at__isnull=True):
        token.revoke()
        count += 1
    return count


# A platform guard: the same name and arity defined twice in one file. This is
# the §3.2 T1 case in Python, and the reason the UID carries an ordinal.
if sys.platform == "win32":
    def clock_skew_seconds(now) -> int:
        return 0
else:
    def clock_skew_seconds(now) -> int:
        return int(now.timestamp()) % 2
