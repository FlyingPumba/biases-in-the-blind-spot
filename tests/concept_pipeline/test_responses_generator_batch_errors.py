"""Tests for ResponsesGenerator batch error handling.

Tests cover:
- All requests fail → raises with full error details printed
- Some requests fail → prints errors, falls back to non-batched generation
- Partial failures print all errors without truncation
"""

import json

import pytest

from biases_in_the_blind_spot.concept_pipeline.responses_generator import (
    ResponsesGenerator,
)


class _MockFile:
    def __init__(self, file_id: str):
        self.id = file_id


class _MockFileContent:
    def __init__(self, text: str):
        self.text = text


class _MockBatchWithCounts:
    """Mock batch object with request_counts."""

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
        self._total = total_requests
        self._error_message = error_message
        self.files = self._MockFiles(self)
        self.batches = self._MockBatches(self)
        self._poll_count = 0

    class _MockFiles:
        def __init__(self, parent: "_MockOpenAIClientAllFailed"):
            self._parent = parent

        def create(self, file, purpose: str):
            return _MockFile("file-123")

        def content(self, file_id: str):
            # Return error file content
            lines = []
            for i in range(self._parent._total):
                lines.append(
                    json.dumps(
                        {
                            "custom_id": f"req_{i}",
                            "response": {
                                "status_code": 400,
                                "body": {
                                    "error": {
                                        "message": self._parent._error_message,
                                        "type": "invalid_request_error",
                                    }
                                },
                            },
                        }
                    )
                )
            return _MockFileContent("\n".join(lines))

    class _MockBatches:
        def __init__(self, parent: "_MockOpenAIClientAllFailed"):
            self._parent = parent

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
            return _MockBatchWithCounts(
                batch_id,
                "completed",
                output_file_id=None,
                error_file_id="error-file-123",
                completed=0,
                failed=self._parent._total,
                total=self._parent._total,
            )


class _MockOpenAIClientPartialFailure:
    """Mock OpenAI client that simulates SOME requests failing."""

    def __init__(
        self,
        total_requests: int,
        failed_indices: list[int],
        error_message: str = "Rate limit exceeded",
    ):
        self._total = total_requests
        self._failed_indices = set(failed_indices)
        self._error_message = error_message
        self.files = self._MockFiles(self)
        self.batches = self._MockBatches(self)
        self._poll_count = 0

    class _MockFiles:
        def __init__(self, parent: "_MockOpenAIClientPartialFailure"):
            self._parent = parent

        def create(self, file, purpose: str):
            return _MockFile("file-123")

        def content(self, file_id: str):
            if file_id == "output-file-789":
                # Return successful responses only
                lines = []
                for i in range(self._parent._total):
                    if i not in self._parent._failed_indices:
                        lines.append(
                            json.dumps(
                                {
                                    "custom_id": f"req_{i}",
                                    "response": {
                                        "body": {
                                            "choices": [
                                                {
                                                    "message": {
                                                        "content": f"response_{i}"
                                                    }
                                                }
                                            ]
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


def test_openai_batch_all_requests_fail_raises(monkeypatch, capsys):
    """Test that when ALL requests in a batch fail, it raises RuntimeError with full error details."""
    mock_client = _MockOpenAIClientAllFailed(
        total_requests=3, error_message="max_tokens is too large: 6000"
    )

    monkeypatch.setattr(
        "biases_in_the_blind_spot.concept_pipeline.responses_generator.OpenAI",
        lambda: mock_client,
    )

    gen = ResponsesGenerator(
        model_name="openai/gpt-4o-mini",
        use_openai_batches=True,
        use_anthropic_batches=False,
        api_batch_poll_seconds=1,
    )

    template = "Test prompt: {x}"
    params = [{"x": "0"}, {"x": "1"}, {"x": "2"}]

    with pytest.raises(RuntimeError) as exc_info:
        gen.generate(template, params)

    # Check that error message mentions all requests failed
    error_msg = str(exc_info.value).lower()
    assert "all" in error_msg or "3" in error_msg
    assert "failed" in error_msg

    # Check that errors were printed to stdout (no truncation)
    captured = capsys.readouterr()
    assert "max_tokens is too large" in captured.out
    # All 3 requests should have their errors printed
    assert "req_0" in captured.out
    assert "req_1" in captured.out
    assert "req_2" in captured.out


def test_openai_batch_partial_failure_fallback(monkeypatch, capsys):
    """Test that when SOME requests fail, it falls back to non-batched generation."""
    mock_client = _MockOpenAIClientPartialFailure(
        total_requests=3,
        failed_indices=[1],  # Only request 1 fails
        error_message="Rate limit exceeded",
    )

    monkeypatch.setattr(
        "biases_in_the_blind_spot.concept_pipeline.responses_generator.OpenAI",
        lambda: mock_client,
    )

    # Track fallback calls
    fallback_called = {"count": 0, "params": None}

    def mock_generate_open_router(self, template, params, temp, max_tokens, batch_size):
        fallback_called["count"] += 1
        fallback_called["params"] = params
        # Return mock responses for failed requests
        return [f"fallback_response_{i}" for i in range(len(params))]

    monkeypatch.setattr(
        ResponsesGenerator,
        "_generate_open_router",
        mock_generate_open_router,
    )

    gen = ResponsesGenerator(
        model_name="openai/gpt-4o-mini",
        use_openai_batches=True,
        use_anthropic_batches=False,
        api_batch_poll_seconds=1,
    )

    template = "Test prompt: {x}"
    params = [{"x": "0"}, {"x": "1"}, {"x": "2"}]

    result = gen.generate(template, params)

    # Should have 3 responses
    assert len(result) == 3
    # Successful requests should have their responses
    assert result[0] == "response_0"
    assert result[2] == "response_2"
    # Failed request should have fallback response
    assert result[1] == "fallback_response_0"

    # Fallback should have been called with the failed request only
    assert fallback_called["count"] == 1
    assert len(fallback_called["params"]) == 1
    assert fallback_called["params"][0] == {"x": "1"}

    # Check that error was printed to stdout
    captured = capsys.readouterr()
    assert "req_1" in captured.out
    assert "Rate limit exceeded" in captured.out
    assert "Falling back" in captured.out


def test_openai_batch_partial_failure_prints_all_errors(monkeypatch, capsys):
    """Test that partial failures print ALL errors without truncation."""
    # Create a batch with many failures
    failed_indices = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]  # 10 failures
    mock_client = _MockOpenAIClientPartialFailure(
        total_requests=20,
        failed_indices=failed_indices,
        error_message="Service temporarily unavailable",
    )

    monkeypatch.setattr(
        "biases_in_the_blind_spot.concept_pipeline.responses_generator.OpenAI",
        lambda: mock_client,
    )

    # Mock the fallback to avoid actual API calls
    def mock_generate_open_router(self, template, params, temp, max_tokens, batch_size):
        return [f"fallback_{i}" for i in range(len(params))]

    monkeypatch.setattr(
        ResponsesGenerator,
        "_generate_open_router",
        mock_generate_open_router,
    )

    gen = ResponsesGenerator(
        model_name="openai/gpt-4o-mini",
        use_openai_batches=True,
        use_anthropic_batches=False,
        api_batch_poll_seconds=1,
    )

    template = "Test prompt: {x}"
    params = [{"x": str(i)} for i in range(20)]

    result = gen.generate(template, params)

    assert len(result) == 20

    # Check that ALL errors were printed (no truncation)
    captured = capsys.readouterr()
    for idx in failed_indices:
        assert f"req_{idx}" in captured.out, f"Error for req_{idx} should be printed"
    assert "Service temporarily unavailable" in captured.out


def test_openai_batch_disables_batching_for_fallback(monkeypatch):
    """Test that fallback generation disables OpenAI batching."""
    mock_client = _MockOpenAIClientPartialFailure(
        total_requests=2,
        failed_indices=[0],
        error_message="Test error",
    )

    monkeypatch.setattr(
        "biases_in_the_blind_spot.concept_pipeline.responses_generator.OpenAI",
        lambda: mock_client,
    )

    # Track the state of batching flags during fallback
    batching_state_during_fallback = {"openai": None, "anthropic": None}

    def mock_generate_open_router(self, template, params, temp, max_tokens, batch_size):
        batching_state_during_fallback["openai"] = self.use_openai_batches
        batching_state_during_fallback["anthropic"] = self.use_anthropic_batches
        return ["fallback_response"]

    monkeypatch.setattr(
        ResponsesGenerator,
        "_generate_open_router",
        mock_generate_open_router,
    )

    gen = ResponsesGenerator(
        model_name="openai/gpt-4o-mini",
        use_openai_batches=True,
        use_anthropic_batches=False,
        api_batch_poll_seconds=1,
    )

    template = "Test: {x}"
    params = [{"x": "0"}, {"x": "1"}]

    gen.generate(template, params)

    # During fallback, batching flags should be False
    assert batching_state_during_fallback["openai"] is False
    assert batching_state_during_fallback["anthropic"] is False

    # After fallback, flags should be restored
    assert gen.use_openai_batches is True
    assert gen.use_anthropic_batches is False


class _MockOpenAIClientOutputFileIdDelayed:
    """Mock client where output_file_id is None on first poll, then available."""

    def __init__(self, total_requests: int):
        self._total = total_requests
        self.files = self._MockFiles(self)
        self.batches = self._MockBatches(self)
        self._poll_count = 0

    class _MockFiles:
        def __init__(self, parent: "_MockOpenAIClientOutputFileIdDelayed"):
            self._parent = parent

        def create(self, file, purpose: str):
            return _MockFile("file-123")

        def content(self, file_id: str):
            lines = []
            for i in range(self._parent._total):
                lines.append(
                    json.dumps(
                        {
                            "custom_id": f"req_{i}",
                            "response": {
                                "body": {
                                    "choices": [
                                        {"message": {"content": f"response_{i}"}}
                                    ]
                                }
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
                # First poll: completed but no output_file_id yet
                return _MockBatchWithCounts(
                    batch_id,
                    "completed",
                    output_file_id=None,
                    completed=self._parent._total,
                    failed=0,
                    total=self._parent._total,
                )
            else:
                # Second poll: output_file_id available
                return _MockBatchWithCounts(
                    batch_id,
                    "completed",
                    output_file_id="output-file-789",
                    completed=self._parent._total,
                    failed=0,
                    total=self._parent._total,
                )


def test_openai_batch_output_file_id_delayed_retries(monkeypatch, capsys):
    """Test that when output_file_id is None but not all failed, it retries on next poll."""
    mock_client = _MockOpenAIClientOutputFileIdDelayed(total_requests=2)

    monkeypatch.setattr(
        "biases_in_the_blind_spot.concept_pipeline.responses_generator.OpenAI",
        lambda: mock_client,
    )

    gen = ResponsesGenerator(
        model_name="openai/gpt-4o-mini",
        use_openai_batches=True,
        use_anthropic_batches=False,
        api_batch_poll_seconds=1,
    )

    template = "Test: {x}"
    params = [{"x": "0"}, {"x": "1"}]

    result = gen.generate(template, params)

    # Should eventually succeed
    assert len(result) == 2
    assert result[0] == "response_0"
    assert result[1] == "response_1"

    # Should have polled at least twice
    assert mock_client._poll_count >= 2

    # Check that retry message was printed
    captured = capsys.readouterr()
    assert "retry" in captured.out.lower() or "output_file_id" in captured.out.lower()
