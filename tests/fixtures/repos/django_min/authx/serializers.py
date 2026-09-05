"""DRF serializers for authentication."""
from rest_framework import serializers

from .models import ApiToken, User


class UserSerializer(serializers.ModelSerializer):
    """Public representation of a user."""

    display_name = serializers.SerializerMethodField()

    class Meta:
        model = User
        fields = ["id", "email", "is_verified", "display_name"]

    def get_display_name(self, obj) -> str:
        return obj.display_name()


class LoginSerializer(serializers.Serializer):
    """Credentials payload."""

    email = serializers.EmailField()
    password = serializers.CharField(write_only=True)

    def validate(self, attrs):
        if not attrs.get("password"):
            raise serializers.ValidationError("password required")
        return attrs


class ApiTokenSerializer(serializers.ModelSerializer):
    class Meta:
        model = ApiToken
        fields = ["key", "revoked_at"]
