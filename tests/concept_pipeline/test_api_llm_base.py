from dataclasses import dataclass

import pytest
from chat_limiter import Message, MessageRole

from biases_in_the_blind_spot.concept_pipeline.api_llm_base import APILLMBase


class _Choice:
    def __init__(self, content: str):
        self.message = type("Msg", (), {"content": content})


class _DummyResponse:
    def __init__(self, success: bool, content: str | None):
        self.success = success
        self.result = (
            type("Res", (), {"choices": [_Choice(content)]})() if content else None
        )
        self.choices = [_Choice(content)] if content else []


class _DummyLimiter:
    def __init__(self, responses):
        self._responses = responses
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def chat_completion(self, **_kwargs):
        idx = len(self.calls)
        self.calls.append(_kwargs)
        return self._responses[idx]


async def _stub_process_batch(_limiter, requests, _config):
    out = []
    for req in requests:
        out.append(_DummyResponse(True, f"ok-{req.messages[0].content}"))
    return out


@dataclass
class _TestLLM(APILLMBase):
    llm_model_name: str = "dummy"
    max_retries_per_item: int = 1


@pytest.mark.anyio("asyncio")
async def test_generate_batch_llm_response(monkeypatch):
    # Patch ChatLimiter.for_model and process_chat_completion_batch
    dummy_limiter = _DummyLimiter([])
    monkeypatch.setattr(
        "biases_in_the_blind_spot.concept_pipeline.api_llm_base.ChatLimiter.for_model",
        lambda *args, **kwargs: dummy_limiter,
    )
    monkeypatch.setattr(
        "biases_in_the_blind_spot.concept_pipeline.api_llm_base.process_chat_completion_batch",
        _stub_process_batch,
    )

    llm = _TestLLM(llm_model_name="x")
    msgs = {
        1: [Message(role=MessageRole.USER, content="m1")],
        2: [Message(role=MessageRole.USER, content="m2")],
    }
    out = await llm._generate_batch_llm_response(msgs)
    assert out[1] == "ok-m1"
    assert out[2] == "ok-m2"


@pytest.mark.anyio("asyncio")
async def test_generate_single_llm_response(monkeypatch):
    resp = _DummyResponse(True, "hello")
    limiter = _DummyLimiter([resp])
    monkeypatch.setattr(
        "biases_in_the_blind_spot.concept_pipeline.api_llm_base.ChatLimiter.for_model",
        lambda *args, **kwargs: limiter,
    )

    llm = _TestLLM(llm_model_name="x")
    msg = Message(role=MessageRole.USER, content="test")
    out = await llm._generate_single_llm_response([msg])
    assert out == "hello"
    assert limiter.calls, "Limiter chat_completion should be called"


class _MockFile:
    def __init__(self, file_id: str):
        self.id = file_id


class _MockBatch:
    def __init__(self, batch_id: str, status: str, output_file_id: str | None = None):
        self.id = batch_id
        self.status = status
        self.output_file_id = output_file_id
        self.error_file_id = None


class _MockFileContent:
    def __init__(self, text: str):
        self.text = text


class _MockOpenAIClient:
    """Mock OpenAI client for testing batch API."""

    def __init__(self, batch_responses: list[dict]):
        self.files = self._MockFiles(self)
        self.batches = self._MockBatches(self)
        self._batch_responses = batch_responses
        self._poll_count = 0
        self._uploaded_content: str = ""

    class _MockFiles:
        def __init__(self, parent: "_MockOpenAIClient"):
            self._parent = parent
            self.created_files: list[str] = []

        def create(self, file, purpose: str):
            self._parent._uploaded_content = file.read().decode()
            return _MockFile("file-123")

        def content(self, file_id: str):
            import json

            lines = []
            for i, resp in enumerate(self._parent._batch_responses):
                lines.append(
                    json.dumps(
                        {
                            "custom_id": f"req_{i}",
                            "response": {
                                "body": {
                                    "choices": [
                                        {"message": {"content": resp["content"]}}
                                    ]
                                }
                            },
                        }
                    )
                )
            return _MockFileContent("\n".join(lines))

    class _MockBatches:
        def __init__(self, parent: "_MockOpenAIClient"):
            self._parent = parent

        def create(
            self,
            input_file_id: str,
            endpoint: str,
            completion_window: str,
            metadata: dict,
        ):
            return _MockBatch("batch-456", "validating")

        def retrieve(self, batch_id: str):
            self._parent._poll_count += 1
            # Return "completed" on first poll for faster tests
            return _MockBatch(batch_id, "completed", "output-file-789")


@pytest.mark.anyio("asyncio")
async def test_generate_batch_llm_response_openai_batches(monkeypatch):
    """Test that OpenAI batch API is used when use_openai_batches=True."""
    batch_responses = [
        {"content": "response-for-key1"},
        {"content": "response-for-key2"},
    ]
    mock_client = _MockOpenAIClient(batch_responses)

    monkeypatch.setattr(
        "biases_in_the_blind_spot.concept_pipeline.api_llm_base.OpenAI",
        lambda **kwargs: mock_client,
    )

    llm = _TestLLM(
        llm_model_name="gpt-4o-mini",  # No prefix needed - uses provider field
        use_openai_batches=True,
        api_batch_poll_seconds=1,
    )

    msgs = {
        "key1": [Message(role=MessageRole.USER, content="prompt1")],
        "key2": [Message(role=MessageRole.USER, content="prompt2")],
    }

    out = await llm._generate_batch_llm_response(msgs)

    assert out["key1"] == "response-for-key1"
    assert out["key2"] == "response-for-key2"
    assert mock_client._poll_count >= 1, "Batch should have been polled"


def test_get_model_id():
    """Test _get_model_id helper."""
    # With provider prefix
    assert APILLMBase._get_model_id("openai/gpt-4o") == "gpt-4o"
    assert APILLMBase._get_model_id("openai:gpt-4o") == "gpt-4o"
    assert APILLMBase._get_model_id("anthropic/claude-3") == "claude-3"
    assert APILLMBase._get_model_id("google/gemma-3") == "gemma-3"

    # Without provider prefix (returned as-is)
    assert APILLMBase._get_model_id("gpt-4o-mini") == "gpt-4o-mini"
    assert APILLMBase._get_model_id("gpt-5-mini") == "gpt-5-mini"


def test_is_transient_batch_poll_error():
    """Test _is_transient_batch_poll_error helper."""
    # Timeout errors should be transient
    assert APILLMBase._is_transient_batch_poll_error(TimeoutError()) is True
    assert APILLMBase._is_transient_batch_poll_error(ConnectionError()) is True

    # Regular errors should not be transient
    assert APILLMBase._is_transient_batch_poll_error(ValueError()) is False

    # 5xx status codes should be transient
    class Error5xx(Exception):
        status_code = 503

    assert APILLMBase._is_transient_batch_poll_error(Error5xx()) is True


def test_config_includes_batch_fields():
    """Test that config property includes batch-related fields."""
    llm = _TestLLM(
        llm_model_name="openai/gpt-4o",
        use_openai_batches=True,
        api_batch_poll_seconds=120,
    )
    cfg = llm.config
    assert cfg["use_openai_batches"] is True
    assert cfg["api_batch_poll_seconds"] == 120


# =============================================================================
# Tests for batch error handling
# =============================================================================


class _MockBatchWithCounts:
    """Mock batch object with request_counts for testing error scenarios."""

    def __init__(
        self,
        batch_id: str,
        status: str,
        output_file_id: str | None = None,
        error_file_id: str | None = None,
        completed: int = 0,
        failed: int = 0,
        total: int = 0,
    ):
        self.id = batch_id
        self.status = status
        self.output_file_id = output_file_id
        self.error_file_id = error_file_id
        self.errors = None
        self.request_counts = type(
            "RequestCounts",
            (),
            {"completed": completed, "failed": failed, "total": total},
        )()


class _MockOpenAIClientAllFailed:
    """Mock OpenAI client that simulates ALL requests failing."""

    def __init__(
        self, total_requests: int, error_message: str = "max_tokens is too large"
    ):
        self.files = self._MockFiles(self, total_requests, error_message)
        self.batches = self._MockBatches(self, total_requests)
        self._poll_count = 0
        self._total_requests = total_requests
        self._error_message = error_message

    class _MockFiles:
        def __init__(
            self, parent: "_MockOpenAIClientAllFailed", total: int, error_msg: str
        ):
            self._parent = parent
            self._total = total
            self._error_msg = error_msg

        def create(self, file, purpose: str):
            return _MockFile("file-123")

        def content(self, file_id: str):
            import json

            # Return error file content
            lines = []
            for i in range(self._total):
                lines.append(
                    json.dumps(
                        {
                            "custom_id": f"req_{i}",
                            "response": {
                                "status_code": 400,
                                "body": {
                                    "error": {
                                        "message": self._error_msg,
                                        "type": "invalid_request_error",
                                    }
                                },
                            },
                        }
                    )
                )
            return _MockFileContent("\n".join(lines))

    class _MockBatches:
        def __init__(self, parent: "_MockOpenAIClientAllFailed", total: int):
            self._parent = parent
            self._total = total

        def create(
            self,
            input_file_id: str,
            endpoint: str,
            completion_window: str,
            metadata: dict,
        ):
            return _MockBatchWithCounts("batch-all-failed", "validating")

        def retrieve(self, batch_id: str):
            self._parent._poll_count += 1
            # Return completed but with no output_file_id (all failed)
            return _MockBatchWithCounts(
                batch_id,
                "completed",
                output_file_id=None,
                error_file_id="error-file-123",
                completed=0,
                failed=self._total,
                total=self._total,
            )


class _MockOpenAIClientPartialFailure:
    """Mock OpenAI client that simulates SOME requests failing."""

    def __init__(
        self,
        success_responses: list[str],
        failed_indices: list[int],
        error_message: str = "Rate limit exceeded",
    ):
        self._success_responses = success_responses
        self._failed_indices = set(failed_indices)
        self._error_message = error_message
        self._total = len(success_responses)
        self.files = self._MockFiles(self)
        self.batches = self._MockBatches(self)
        self._poll_count = 0

    class _MockFiles:
        def __init__(self, parent: "_MockOpenAIClientPartialFailure"):
            self._parent = parent

        def create(self, file, purpose: str):
            return _MockFile("file-123")

        def content(self, file_id: str):
            import json

            if file_id == "output-file-789":
                # Return successful responses only
                lines = []
                for i, resp in enumerate(self._parent._success_responses):
                    if i not in self._parent._failed_indices:
                        lines.append(
                            json.dumps(
                                {
                                    "custom_id": f"req_{i}",
                                    "response": {
                                        "body": {
                                            "choices": [{"message": {"content": resp}}]
                                        }
                                    },
                                }
                            )
                        )
                return _MockFileContent("\n".join(lines))
            elif file_id == "error-file-456":
                # Return error file for failed requests
                lines = []
                for i in self._parent._failed_indices:
                    lines.append(
                        json.dumps(
                            {
                                "custom_id": f"req_{i}",
                                "response": {
                                    "status_code": 429,
                                    "body": {
                                        "error": {
                                            "message": self._parent._error_message,
                                            "type": "rate_limit_error",
                                        }
                                    },
                                },
                            }
                        )
                    )
                return _MockFileContent("\n".join(lines))
            return _MockFileContent("")

    class _MockBatches:
        def __init__(self, parent: "_MockOpenAIClientPartialFailure"):
            self._parent = parent

        def create(
            self,
            input_file_id: str,
            endpoint: str,
            completion_window: str,
            metadata: dict,
        ):
            return _MockBatchWithCounts("batch-partial", "validating")

        def retrieve(self, batch_id: str):
            self._parent._poll_count += 1
            num_failed = len(self._parent._failed_indices)
            num_success = self._parent._total - num_failed
            return _MockBatchWithCounts(
                batch_id,
                "completed",
                output_file_id="output-file-789",
                error_file_id="error-file-456",
                completed=num_success,
                failed=num_failed,
                total=self._parent._total,
            )


@pytest.mark.anyio("asyncio")
async def test_openai_batch_all_requests_fail_raises(monkeypatch, capsys):
    """Test that when ALL requests in a batch fail, it raises RuntimeError with full error details."""
    mock_client = _MockOpenAIClientAllFailed(
        total_requests=3, error_message="max_tokens is too large: 6000"
    )

    monkeypatch.setattr(
        "biases_in_the_blind_spot.concept_pipeline.api_llm_base.OpenAI",
        lambda **kwargs: mock_client,
    )

    llm = _TestLLM(
        llm_model_name="gpt-4o-mini",
        use_openai_batches=True,
        api_batch_poll_seconds=1,
    )

    msgs = {
        "key0": [Message(role=MessageRole.USER, content="prompt0")],
        "key1": [Message(role=MessageRole.USER, content="prompt1")],
        "key2": [Message(role=MessageRole.USER, content="prompt2")],
    }

    with pytest.raises(RuntimeError) as exc_info:
        await llm._generate_batch_llm_response(msgs)

    # Check that error message mentions all requests failed
    assert "all" in str(exc_info.value).lower()
    assert "failed" in str(exc_info.value).lower()

    # Check that errors were printed to stdout
    captured = capsys.readouterr()
    assert "max_tokens is too large" in captured.out
    # All 3 requests should have their errors printed
    assert "req_0" in captured.out
    assert "req_1" in captured.out
    assert "req_2" in captured.out


@pytest.mark.anyio("asyncio")
async def test_openai_batch_partial_failure_continues(monkeypatch, capsys):
    """Test that when SOME requests fail, errors are printed and empty strings stored."""
    mock_client = _MockOpenAIClientPartialFailure(
        success_responses=["response0", "response1", "response2"],
        failed_indices=[1],  # Only request 1 fails
        error_message="Rate limit exceeded",
    )

    monkeypatch.setattr(
        "biases_in_the_blind_spot.concept_pipeline.api_llm_base.OpenAI",
        lambda **kwargs: mock_client,
    )

    llm = _TestLLM(
        llm_model_name="gpt-4o-mini",
        use_openai_batches=True,
        api_batch_poll_seconds=1,
    )

    msgs = {
        "key0": [Message(role=MessageRole.USER, content="prompt0")],
        "key1": [Message(role=MessageRole.USER, content="prompt1")],
        "key2": [Message(role=MessageRole.USER, content="prompt2")],
    }

    out = await llm._generate_batch_llm_response(msgs)

    # Successful requests should have their responses
    assert out["key0"] == "response0"
    assert out["key2"] == "response2"
    # Failed request should have empty string
    assert out["key1"] == ""

    # Check that error was printed to stdout
    captured = capsys.readouterr()
    assert "req_1" in captured.out
    assert "Rate limit exceeded" in captured.out


class _MockOpenAIClientOutputFileIdDelayed:
    """Mock client where output_file_id is None on first poll, then available."""

    def __init__(self, responses: list[str]):
        self._responses = responses
        self.files = self._MockFiles(self)
        self.batches = self._MockBatches(self)
        self._poll_count = 0

    class _MockFiles:
        def __init__(self, parent: "_MockOpenAIClientOutputFileIdDelayed"):
            self._parent = parent

        def create(self, file, purpose: str):
            return _MockFile("file-123")

        def content(self, file_id: str):
            import json

            lines = []
            for i, resp in enumerate(self._parent._responses):
                lines.append(
                    json.dumps(
                        {
                            "custom_id": f"req_{i}",
                            "response": {
                                "body": {"choices": [{"message": {"content": resp}}]}
                            },
                        }
                    )
                )
            return _MockFileContent("\n".join(lines))

    class _MockBatches:
        def __init__(self, parent: "_MockOpenAIClientOutputFileIdDelayed"):
            self._parent = parent

        def create(
            self,
            input_file_id: str,
            endpoint: str,
            completion_window: str,
            metadata: dict,
        ):
            return _MockBatchWithCounts("batch-delayed", "validating")

        def retrieve(self, batch_id: str):
            self._parent._poll_count += 1
            if self._parent._poll_count == 1:
                # First poll: completed but no output_file_id yet (race condition)
                return _MockBatchWithCounts(
                    batch_id,
                    "completed",
                    output_file_id=None,
                    completed=len(self._parent._responses),
                    failed=0,
                    total=len(self._parent._responses),
                )
            else:
                # Second poll: output_file_id is available
                return _MockBatchWithCounts(
                    batch_id,
                    "completed",
                    output_file_id="output-file-789",
                    completed=len(self._parent._responses),
                    failed=0,
                    total=len(self._parent._responses),
                )


@pytest.mark.anyio("asyncio")
async def test_openai_batch_output_file_id_delayed_retries(monkeypatch, capsys):
    """Test that when output_file_id is None but not all failed, it retries on next poll."""
    mock_client = _MockOpenAIClientOutputFileIdDelayed(
        responses=["response0", "response1"]
    )

    monkeypatch.setattr(
        "biases_in_the_blind_spot.concept_pipeline.api_llm_base.OpenAI",
        lambda **kwargs: mock_client,
    )

    llm = _TestLLM(
        llm_model_name="gpt-4o-mini",
        use_openai_batches=True,
        api_batch_poll_seconds=1,  # Short poll for fast test
    )

    msgs = {
        "key0": [Message(role=MessageRole.USER, content="prompt0")],
        "key1": [Message(role=MessageRole.USER, content="prompt1")],
    }

    out = await llm._generate_batch_llm_response(msgs)

    # Should eventually succeed after retry
    assert out["key0"] == "response0"
    assert out["key1"] == "response1"
    # Should have polled at least twice
    assert mock_client._poll_count >= 2

    # Check that retry message was printed
    captured = capsys.readouterr()
    assert "output_file_id" in captured.out.lower() or "retry" in captured.out.lower()
