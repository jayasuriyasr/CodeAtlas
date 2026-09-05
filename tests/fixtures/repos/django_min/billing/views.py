"""Billing endpoints."""
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from common.utils import audit_event
from .models import Subscription
from .tasks import charge_card


class SubscriptionView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request) -> Response:
        subs = Subscription.objects.filter(user=request.user, active=True)
        return Response([{"plan": s.plan.code, "cents": s.monthly_total()} for s in subs])

    def post(self, request) -> Response:
        sub = Subscription.objects.get(pk=request.data["id"])
        charge_card(sub.user, sub.monthly_total())
        audit_event("charge", user_id=sub.user.pk)
        return Response({"charged": sub.monthly_total()})
