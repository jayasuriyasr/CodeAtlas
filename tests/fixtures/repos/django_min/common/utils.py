"""Small shared helpers. Deliberately hub-like: many modules call these."""
import hmac
import logging
import re

logger = logging.getLogger(__name__)

_EMAIL = re.compile(r"^([^@]+)@(.+)$")


def constant_time_equals(left: str, right: str) -> bool:
    """Compare two strings without leaking length information via timing."""
    return hmac.compare_digest(left.encode(), right.encode())


def mask_email(email: str) -> str:
    """Reduce an email to its first character plus domain, for logs."""
    match = _EMAIL.match(email or "")
    if match is None:
        return "<invalid>"
    local, domain = match.groups()
    return f"{local[0]}***@{domain}"


def audit_event(name: str, **fields) -> None:
    """Emit one structured audit line."""
    logger.info("audit %s %s", name, fields)


def chunked(items, size: int):
    """Yield successive slices of `items` of at most `size`."""
    buf = []
    for item in items:
        buf.append(item)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def get_user_by_id(user_id: int):
    """Snake-case name, paired with getUserById in the TS fixture (§5.3)."""
    from authx.models import User

    return User.objects.filter(pk=user_id).first()
