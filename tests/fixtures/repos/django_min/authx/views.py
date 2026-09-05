"""Authentication views."""
import logging
from typing import TYPE_CHECKING

from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from common.utils import audit_event, mask_email
from .models import ApiToken, User
from .serializers import LoginSerializer, UserSerializer
from .services import issue_token, verify_credentials

if TYPE_CHECKING:
    from rest_framework.request import Request

logger = logging.getLogger(__name__)


class HealthView(APIView):
    """Liveness probe."""

    permission_classes = [AllowAny]

    def get(self, request) -> Response:
        return Response({"ok": True}, status=status.HTTP_200_OK)


class LoginView(APIView):
    """Exchange credentials for an API token."""

    permission_classes = [AllowAny]

    def post(self, request) -> Response:
        """Validate credentials and issue a token."""
        serializer = LoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = verify_credentials(
            serializer.validated_data["email"],
            serializer.validated_data["password"],
        )
        if user is None:
            logger.warning("login failed for %s", mask_email(serializer.validated_data["email"]))
            return Response({"detail": "invalid"}, status=status.HTTP_401_UNAUTHORIZED)
        token = issue_token(user)
        audit_event("login", user_id=user.pk)
        return Response({"token": token.key}, status=status.HTTP_201_CREATED)


class ProfileView(APIView):
    """Read and update the current user."""

    permission_classes = [IsAuthenticated]

    def get(self, request) -> Response:
        return Response(UserSerializer(request.user).data)

    def patch(self, request) -> Response:
        serializer = UserSerializer(request.user, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        audit_event("profile_update", user_id=request.user.pk)
        return Response(serializer.data)


class TokenRevokeView(APIView):
    permission_classes = [IsAuthenticated]

    def delete(self, request, key: str) -> Response:
        token = ApiToken.objects.filter(user=request.user, key=key).first()
        if token is None:
            return Response(status=status.HTTP_404_NOT_FOUND)
        token.revoke()
        audit_event("token_revoke", user_id=request.user.pk)
        return Response(status=status.HTTP_204_NO_CONTENT)


def _lookup_user(email: str) -> "User | None":
    """Module-level helper; deliberately not a method."""
    return User.objects.filter(email__iexact=email).first()
