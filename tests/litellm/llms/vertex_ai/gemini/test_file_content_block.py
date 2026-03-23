"""
Tests for handling malformed 'file' content blocks (missing 'file' sub-field).

Regression tests for:
- litellm/llms/vertex_ai/gemini/transformation.py
- litellm/llms/gemini/chat/transformation.py
- litellm/litellm_core_utils/prompt_templates/common_utils.py
- litellm/litellm_core_utils/prompt_templates/factory.py (Bedrock)
- litellm/llms/openai/chat/gpt_transformation.py
"""

import asyncio
import os
import sys
from typing import List, cast

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    get_file_ids_from_messages,
    update_messages_with_model_file_ids,
)
from litellm.litellm_core_utils.prompt_templates.factory import (
    BedrockConverseMessagesProcessor,
)
from litellm.llms.vertex_ai.gemini.transformation import (
    _gemini_convert_messages_with_history,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionFileObject,
    OpenAIMessageContentListBlock,
)

_MALFORMED_MESSAGES_RAW = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "hello"},
            {"type": "file"},  # Missing required "file" sub-field
        ],
    }
]

_WELL_FORMED_MESSAGES_RAW = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "hello"},
            {
                "type": "file",
                "file": {"file_id": "file-abc123", "format": "pdf"},
            },
        ],
    }
]

MALFORMED_FILE_OBJECT: ChatCompletionFileObject = cast(
    ChatCompletionFileObject, {"type": "file"}
)


def _malformed() -> List[AllMessageValues]:
    return cast(List[AllMessageValues], _MALFORMED_MESSAGES_RAW)


def _well_formed() -> List[AllMessageValues]:
    return cast(List[AllMessageValues], _WELL_FORMED_MESSAGES_RAW)


# ---------------------------------------------------------------------------
# vertex_ai/gemini/transformation.py
# ---------------------------------------------------------------------------


def test_gemini_convert_messages_malformed_file_raises_bad_request():
    """_gemini_convert_messages_with_history should raise BadRequestError (not KeyError)
    when a content block has type='file' but no 'file' sub-field."""
    with pytest.raises(litellm.BadRequestError, match="missing the required 'file' field"):
        _gemini_convert_messages_with_history(
            messages=_malformed(),
            model="gemini-2.0-flash",
        )


def test_gemini_convert_messages_well_formed_file_does_not_raise():
    """_gemini_convert_messages_with_history should not raise a KeyError for well-formed
    file blocks — only provider-level errors (BadRequestError/Exception) are acceptable."""
    try:
        _gemini_convert_messages_with_history(
            messages=_well_formed(),
            model="gemini-2.0-flash",
        )
    except (litellm.BadRequestError, Exception) as e:
        # Provider errors about unresolvable mime type are acceptable.
        # A KeyError surfacing as "missing the required 'file' field" is NOT.
        assert "missing the required 'file' field" not in str(e)


# ---------------------------------------------------------------------------
# common_utils.py - update_messages_with_model_file_ids
# ---------------------------------------------------------------------------


def test_update_messages_with_model_file_ids_malformed_skips():
    """update_messages_with_model_file_ids should silently skip content blocks
    that have type='file' but no 'file' sub-field (no KeyError)."""
    result = update_messages_with_model_file_ids(
        messages=_malformed(),
        model_id="some-model",
        model_file_id_mapping={},
    )
    assert result is not None


def test_update_messages_with_model_file_ids_well_formed_updates():
    """update_messages_with_model_file_ids should update file_id for well-formed blocks."""
    mapping = {"file-abc123": {"some-model": "provider-file-xyz"}}
    result = update_messages_with_model_file_ids(
        messages=_well_formed(),
        model_id="some-model",
        model_file_id_mapping=mapping,
    )
    content = result[0].get("content")
    assert isinstance(content, list)
    file_block = next(c for c in content if c.get("type") == "file")
    assert file_block.get("file", {}).get("file_id") == "provider-file-xyz"


# ---------------------------------------------------------------------------
# common_utils.py - get_file_ids_from_messages
# ---------------------------------------------------------------------------


def test_get_file_ids_from_messages_malformed_returns_empty():
    """get_file_ids_from_messages should return [] for malformed file blocks (no KeyError)."""
    result = get_file_ids_from_messages(messages=_malformed())
    assert result == []


def test_get_file_ids_from_messages_well_formed_returns_ids():
    """get_file_ids_from_messages should extract file_id from well-formed blocks."""
    messages: List[AllMessageValues] = cast(
        List[AllMessageValues],
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "file", "file": {"file_id": "file-abc123", "format": "pdf"}},
                ],
            }
        ],
    )
    result = get_file_ids_from_messages(messages=messages)
    assert result == ["file-abc123"]


# ---------------------------------------------------------------------------
# factory.py - BedrockConverseMessagesProcessor (sync + async)
# ---------------------------------------------------------------------------


def test_bedrock_process_file_message_malformed_raises_bad_request():
    """_process_file_message should raise BadRequestError (not KeyError)
    when the file object is missing the 'file' sub-field."""
    with pytest.raises(litellm.BadRequestError, match="missing the required 'file' field"):
        BedrockConverseMessagesProcessor._process_file_message(MALFORMED_FILE_OBJECT)


def test_bedrock_async_process_file_message_malformed_raises_bad_request():
    """_async_process_file_message should raise BadRequestError (not KeyError)
    when the file object is missing the 'file' sub-field."""

    async def _run() -> None:
        await BedrockConverseMessagesProcessor._async_process_file_message(
            MALFORMED_FILE_OBJECT
        )

    with pytest.raises(litellm.BadRequestError, match="missing the required 'file' field"):
        asyncio.run(_run())


# ---------------------------------------------------------------------------
# openai/chat/gpt_transformation.py
# ---------------------------------------------------------------------------


def test_openai_apply_common_transform_malformed_file_raises_bad_request():
    """_apply_common_transform_content_item should raise BadRequestError (not KeyError)
    when a content block has type='file' but no 'file' sub-field."""
    from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig

    config = OpenAIGPTConfig()
    malformed_block: OpenAIMessageContentListBlock = cast(
        OpenAIMessageContentListBlock, {"type": "file"}
    )
    with pytest.raises(litellm.BadRequestError, match="missing the required 'file' field"):
        config._apply_common_transform_content_item(malformed_block)


def test_openai_apply_common_transform_well_formed_file_does_not_raise():
    """_apply_common_transform_content_item should not raise for well-formed file blocks."""
    from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig

    config = OpenAIGPTConfig()
    well_formed_block: OpenAIMessageContentListBlock = cast(
        OpenAIMessageContentListBlock,
        {"type": "file", "file": {"file_id": "file-abc123"}},
    )
    result = config._apply_common_transform_content_item(well_formed_block)
    assert result.get("type") == "file"
    file_field = cast(ChatCompletionFileObject, result).get("file", {})
    assert file_field.get("file_id") == "file-abc123"
