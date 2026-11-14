"""Multi-hop answers reach the reader with their paths and their limits."""

from __future__ import annotations

from .pipeline_support import PY_REPO, ask


def test_each_result_carries_its_call_path(indexed):
    _r, _seeds, _n, notes = ask(indexed, PY_REPO, "what eventually calls `charge_card`")
    structural = notes["structural"]
    assert notes["route_taken"] == "STRUCTURAL" and structural["rule"] == "impact"
    related = {r["qualified_name"]: r for r in structural["related"]}
    assert {"billing.tasks.run_monthly_billing", "billing.tasks.retry_failed",
            "billing.views.SubscriptionView.post"} <= set(related)
    assert related["billing.tasks.retry_failed"]["path"] == [
        "billing.tasks.retry_failed", "billing.tasks.charge_card"]
    assert structural["depth_limit"] >= 1 and structural["truncated"] is False


def test_the_cli_prints_the_path(indexed, capsys):
    from api.cli import _print_notes

    _r, _seeds, _n, notes = ask(indexed, PY_REPO, "what is affected if I change `charge_card`")
    _print_notes(notes)
    err = capsys.readouterr().err
    assert "billing.tasks.retry_failed → billing.tasks.charge_card" in err
    assert "within" in err and "hops" in err
