"""Exercise the real request schema through Pydantic, HTTP, and msgpack."""

from typing import Annotated

import msgspec
import pytest
from fastapi import Body, FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import TypeAdapter, ValidationError

from sglang.srt.managers.io_struct import SetInternalStateReq
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def client_for(request_type):
    app = FastAPI()
    seen = []

    @app.exception_handler(RequestValidationError)
    async def invalid(request, error):
        # Production http_server also maps request validation errors to 400.
        return JSONResponse(status_code=400, content={"message": str(error)})

    async def endpoint(obj):
        seen.append(obj)
        return [True]

    endpoint.__annotations__["obj"] = Annotated[request_type, Body()]
    app.post("/set_internal_state")(endpoint)
    return TestClient(app), seen


def test_http_accepts_reset_and_clear_and_survives_ipc():
    client, seen = client_for(SetInternalStateReq)
    result = client.post(
        "/set_internal_state",
        json={
            "server_args": {
                "dspark_force_budget_frac": None,
                "dspark_clear_info_records": True,
            }
        },
    )
    assert result.status_code == 200 and result.json() == [True]
    assert seen[0].server_args == {
        "dspark_force_budget_frac": None,
        "dspark_clear_info_records": 1,
    }
    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(seen[0]), type=SetInternalStateReq
    )
    assert decoded.server_args == seen[0].server_args


@pytest.mark.parametrize(
    "values",
    [
        {"dspark_force_budget_frac": 0.25},
        {"dspark_force_budget_frac": 1},
        {"dspark_clear_info_records": True},
        {"dspark_clear_info_records": False},
        {"pp_max_micro_batch_size": 2},
        {"speculative_accept_threshold_single": 0.5},
        {"speculative_accept_threshold_acc": 0.9},
    ],
)
def test_existing_numeric_http_behavior_is_preserved(values):
    payload = {"server_args": values}
    assert (
        TypeAdapter(SetInternalStateReq).validate_python(payload).server_args == values
    )


@pytest.mark.parametrize(
    "name",
    [
        "pp_max_micro_batch_size",
        "speculative_accept_threshold_single",
        "speculative_accept_threshold_acc",
        "dspark_clear_info_records",
        "unknown_key",
    ],
)
def test_null_remains_invalid_for_other_keys(name):
    client, seen = client_for(SetInternalStateReq)
    assert (
        client.post(
            "/set_internal_state", json={"server_args": {name: None}}
        ).status_code
        == 400
    )
    assert seen == []


@pytest.mark.parametrize("value", [[], {}, "not-a-number"])
def test_invalid_budget_types_remain_rejected(value):
    with pytest.raises(ValidationError):
        TypeAdapter(SetInternalStateReq).validate_python(
            {"server_args": {"dspark_force_budget_frac": value}}
        )
