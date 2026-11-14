"""A network bind says, out loud, that it is plain HTTP.

Reproduced first: `serve_security("0.0.0.0", env_token=…)` accepted the bind
and the server started on plain HTTP — the bearer token and the source code
cross the network unencrypted, and nothing said so. SEI has no TLS of its own;
the remedy is an SSH tunnel or a TLS proxy in front, which the README
describes. The server now prints that when it binds beyond this machine.
"""

from __future__ import annotations

import pytest

from api.cli import network_bind_warning


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.20", "sei.example.com"])
def test_a_network_bind_warns_that_it_is_plain_http(host):
    warning = network_bind_warning(host)
    assert warning is not None
    assert "plain HTTP" in warning and "token" in warning
    assert "SSH tunnel" in warning and "TLS" in warning


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_a_local_bind_needs_no_warning(host):
    assert network_bind_warning(host) is None
