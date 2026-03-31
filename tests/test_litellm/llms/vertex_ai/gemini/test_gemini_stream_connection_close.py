import asyncio
from typing import Any, Dict

import pytest

from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    ModelResponseIterator,
)


class _FakeAsyncLines:
    """
    Minimal async iterator to simulate httpx.Response.aiter_lines().
    Tracks whether aclose() was called.
    """

    def __init__(self) -> None:
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.closed = True


class _FakeResponse:
    """
    Minimal httpx.Response‑like object for testing.
    Only implements the pieces ModelResponseIterator.aclose/close uses.
    """

    def __init__(self) -> None:
        self.status_code = 200
        self.headers: Dict[str, str] = {}
        self.text = ""
        self.is_closed = False
        self.aclose_called = False
        self.close_called = False

    def aiter_lines(self) -> _FakeAsyncLines:
        return _FakeAsyncLines()

    async def aclose(self) -> None:
        self.aclose_called = True
        self.is_closed = True

    def close(self) -> None:
        self.close_called = True
        self.is_closed = True


class _DummyLoggingObj:
    """
    Minimal logging_obj compatible with CustomStreamWrapper.
    """

    def __init__(self) -> None:
        self.model_call_details: Dict[str, Any] = {"litellm_params": {}}
        self.optional_params: Dict[str, Any] = {}


@pytest.mark.asyncio
async def test_model_response_iterator_aclose_closes_response_and_stream():
    """
    Verify that ModelResponseIterator.aclose() is invoked by CustomStreamWrapper.aclose()
    and that it closes both the async line iterator and the underlying response.
    """

    fake_response = _FakeResponse()
    streaming_response = fake_response.aiter_lines()
    logging_obj = _DummyLoggingObj()

    iterator = ModelResponseIterator(
        streaming_response=streaming_response,
        sync_stream=False,
        logging_obj=logging_obj,  # type: ignore[arg-type]
        response=fake_response,
    )

    wrapper = CustomStreamWrapper(
        completion_stream=iterator,
        model="vertex-gemini-test",
        logging_obj=logging_obj,  # type: ignore[arg-type]
        custom_llm_provider="vertex_ai",
    )

    # Call aclose and ensure it does not raise
    await wrapper.aclose()

    # The streaming_response (aiter_lines generator) must be closed even when
    # iteration never started (early-abort / pre-__aiter__ path).
    assert streaming_response.closed, (
        "streaming_response (aiter_lines) was not closed by aclose()"
    )

    # The underlying HTTP response must also have been closed.
    assert iterator.response is fake_response
    assert fake_response.aclose_called or fake_response.close_called

