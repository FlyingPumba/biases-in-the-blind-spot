import asyncio
import json
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Any

from chat_limiter import (
    BatchConfig,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatLimiter,
    Message,
    process_chat_completion_batch,
)
from openai import OpenAI


@dataclass
class APILLMBase:
    llm_model_name: str
    provider: str | None = None
    api_key: str | None = None
    chat_timeout: float = 600.0
    max_tokens: int = 6000
    temperature: float = 0.7
    max_concurrent: int = 100
    max_retries_per_item: int = 10
    # If True, use OpenAI's Batch API for ~50% cost savings (synchronous polling).
    use_openai_batches: bool = False
    # Poll interval (seconds) for OpenAI batch API. Default is 3 minutes.
    api_batch_poll_seconds: int = 180
    # Max requests per OpenAI batch file. OpenAI has a file size limit (~200MB).
    openai_batch_size: int = 5000
    # Limit how many API calls are scheduled in one asyncio.gather to avoid huge task graphs
    _limiter: ChatLimiter | None = None

    @property
    def config(self) -> dict[str, Any]:
        """Return configuration relevant for API-based LLM classes.

        Note: Intentionally excludes sensitive fields like api_key.
        """
        return {
            "llm_model_name": self.llm_model_name,
            "chat_timeout": self.chat_timeout,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "max_concurrent": self.max_concurrent,
            "use_openai_batches": self.use_openai_batches,
            "api_batch_poll_seconds": self.api_batch_poll_seconds,
        }

    @property
    def limiter(self) -> ChatLimiter:
        if self._limiter is None:
            self._limiter = ChatLimiter.for_model(
                self.llm_model_name, api_key=self.api_key, timeout=self.chat_timeout
            )
        return self._limiter

    @staticmethod
    def _get_model_id(model_name: str) -> str:
        """Get the model ID, stripping any provider prefix if present.

        Examples:
            'openai/gpt-4o' -> 'gpt-4o'
            'anthropic:claude-3' -> 'claude-3'
            'gpt-4o-mini' -> 'gpt-4o-mini' (no prefix, returned as-is)
        """
        assert isinstance(model_name, str) and len(model_name) > 0
        # Check for common provider prefixes and strip them
        for sep in ["/", ":"]:
            if sep in model_name:
                prefix, model_id = model_name.split(sep, 1)
                # Only strip if prefix looks like a provider name
                if prefix.lower() in {
                    "openai",
                    "anthropic",
                    "google",
                    "mistral",
                    "mistralai",
                }:
                    return model_id
        # No prefix found, return as-is
        return model_name

    @staticmethod
    def _is_transient_batch_poll_error(e: Exception) -> bool:
        """Return True for errors that are plausibly transient during batch polling.

        We intentionally keep this heuristic minimal and dependency-free:
        - Treat HTTP 5xx as transient (provider out-of-service, etc.).
        - Treat timeouts/connection errors as transient.
        """
        status_code = getattr(e, "status_code", None)
        if isinstance(status_code, int) and status_code >= 500:
            return True

        response = getattr(e, "response", None)
        resp_status = getattr(response, "status_code", None)
        if isinstance(resp_status, int) and resp_status >= 500:
            return True

        if isinstance(e, TimeoutError | ConnectionError):
            return True

        # Handle common http client exception shapes without importing them.
        name = type(e).__name__.lower()
        mod = type(e).__module__.lower()
        if "timeout" in name or "timeout" in mod:
            return True
        if "connect" in name or "connection" in name:
            return True

        return False

    def _poll_openai_batches(
        self,
        client: OpenAI,
        submitted: list[tuple[str, int, int]],
        total_items: int,
        keys: list[Any],
        model_id: str,
        poll_seconds: int,
    ) -> dict[Any, str | None]:
        # Poll all pending batches until completion
        print(
            f"[APILLMBase] Submitted {len(submitted)} batch(es). Polling until complete..."
        )
        pending: dict[str, tuple[int, int]] = {bid: (s, e) for bid, s, e in submitted}

        responses: dict[int, str] = {}

        while len(pending) > 0:
            status_counts: dict[str, int] = {}
            completed_now: list[str] = []

            for batch_id, _ in list(pending.items()):
                try:
                    batch = client.batches.retrieve(batch_id)
                except Exception as e:
                    if self._is_transient_batch_poll_error(e):
                        print(
                            f"[APILLMBase] Transient error retrieving batch "
                            f"(treating as in_progress): batch_id={batch_id}, "
                            f"error={type(e).__name__}: {e}"
                        )
                        status_counts["in_progress"] = (
                            status_counts.get("in_progress", 0) + 1
                        )
                        continue
                    raise

                status = str(batch.status)
                status_counts[status] = status_counts.get(status, 0) + 1

                if status not in {"completed", "failed", "expired", "cancelled"}:
                    continue

                # Batch is done (status is completed/failed/expired/cancelled)
                if status != "completed":
                    # Try to get error details
                    error_details = ""
                    if hasattr(batch, "errors") and batch.errors:
                        error_details = f", errors={batch.errors}"
                    if batch.error_file_id:
                        try:
                            err_content = client.files.content(batch.error_file_id)
                            error_details += (
                                f", error_file_content={err_content.text[:2000]}"
                            )
                        except Exception:
                            error_details += f", error_file_id={batch.error_file_id}"
                    raise RuntimeError(
                        f"OpenAI batch did not complete successfully: batch_id={batch_id}, "
                        f"status={status}, model={model_id}{error_details}"
                    )

                # Handle case where output_file_id is missing
                if batch.output_file_id is None:
                    # Check if all requests failed
                    req_counts = getattr(batch, "request_counts", None)
                    all_failed = (
                        req_counts is not None
                        and getattr(req_counts, "completed", 0) == 0
                        and getattr(req_counts, "failed", 0) > 0
                    )

                    if all_failed:
                        # Print ALL errors to stdout without truncation
                        print(
                            f"\n[APILLMBase] ERROR: OpenAI batch {batch_id} failed - "
                            f"all {getattr(req_counts, 'total', '?')} requests failed. "
                            f"model={model_id}"
                        )
                        print(f"[APILLMBase] request_counts={req_counts}")
                        first_error_msg = None
                        if batch.error_file_id:
                            try:
                                err_content = client.files.content(batch.error_file_id)
                                print(f"[APILLMBase] Errors from batch {batch_id}:")
                                for line in err_content.text.splitlines():
                                    try:
                                        err_obj = json.loads(line)
                                        custom_id = err_obj.get("custom_id", "?")
                                        resp = err_obj.get("response", {})
                                        body = resp.get("body", {})
                                        error = body.get("error", {})
                                        error_msg = error.get("message", str(error))
                                        error_type = error.get("type", "unknown")
                                        print(
                                            f"  {custom_id}: [{error_type}] {error_msg}"
                                        )
                                        if first_error_msg is None:
                                            first_error_msg = error_msg
                                    except json.JSONDecodeError:
                                        print(f"  (failed to parse error line): {line}")
                            except Exception as e:
                                print(
                                    f"  (could not read error file {batch.error_file_id}: {e})"
                                )
                        raise RuntimeError(
                            f"OpenAI batch failed: all {getattr(req_counts, 'total', '?')} "
                            f"requests failed. batch_id={batch_id}, model={model_id}. "
                            f"First error: {first_error_msg or 'unknown'}"
                        )

                    # output_file_id not ready yet, will retry on next poll
                    print(
                        f"[APILLMBase] Batch {batch_id} completed but output_file_id "
                        f"not ready, will retry. request_counts={req_counts}"
                    )
                    continue

                try:
                    # Process successful responses from output file
                    file_response = client.files.content(batch.output_file_id)
                    for line in file_response.text.splitlines():
                        obj = json.loads(line)
                        custom_id = obj.get("custom_id")
                        assert isinstance(custom_id, str) and len(custom_id) > 0

                        if obj.get("error") is not None:
                            raise RuntimeError(
                                f"OpenAI batch request failed: batch_id={batch_id}, "
                                f"custom_id={custom_id}, error={obj['error']}"
                            )

                        resp_body = obj.get("response", {}).get("body", {})
                        choices = resp_body.get("choices", [])
                        if not choices or "message" not in choices[0]:
                            raise RuntimeError(
                                f"OpenAI batch request missing choices/message: "
                                f"batch_id={batch_id}, custom_id={custom_id}, "
                                f"response={obj}"
                            )

                        assert custom_id.startswith("req_")
                        idx = int(custom_id.split("_", 1)[1])
                        assert 0 <= idx < total_items
                        # Accept empty content (can happen with reasoning models like gpt-5-mini)
                        content = choices[0]["message"].get("content") or ""
                        responses[idx] = content

                    # Also check error file for failed requests within this batch
                    if batch.error_file_id:
                        err_response = client.files.content(batch.error_file_id)
                        for line in err_response.text.splitlines():
                            obj = json.loads(line)
                            custom_id = obj.get("custom_id", "?")
                            error_info = obj.get("error") or obj.get(
                                "response", {}
                            ).get("body", {}).get("error", {})
                            print(
                                f"[APILLMBase] Request {custom_id} failed in batch "
                                f"{batch_id}: {error_info}"
                            )
                            # Store empty string for failed requests so we don't have missing keys
                            if custom_id.startswith("req_"):
                                idx = int(custom_id.split("_", 1)[1])
                                if idx not in responses:
                                    responses[idx] = ""
                except Exception as e:
                    if self._is_transient_batch_poll_error(e):
                        print(
                            f"[APILLMBase] Transient error processing batch "
                            f"(treating as in_progress): batch_id={batch_id}, "
                            f"error={type(e).__name__}: {e}"
                        )
                        continue
                    raise

                del pending[batch_id]
                completed_now.append(batch_id)

            print(
                f"[APILLMBase] Batches pending={len(pending)} "
                f"(just_completed={len(completed_now)}): {status_counts}"
            )
            if len(pending) > 0:
                time.sleep(poll_seconds)

        # Validate we got all responses
        missing_indices = [i for i in range(total_items) if i not in responses]
        if missing_indices:
            print(
                f"[APILLMBase] WARNING: {len(missing_indices)} responses missing "
                f"(first 10: {missing_indices[:10]}). Filling with empty strings."
            )
            for idx in missing_indices:
                responses[idx] = ""

        # Map responses back to keys
        key_to_response: dict[Any, str | None] = {}
        for idx, key in enumerate(keys):
            key_to_response[key] = responses.get(idx)

        return key_to_response

    def _generate_batch_llm_response_openai_from_batch_ids(
        self,
        key_to_messages: dict[Any, list[Message]],
        batch_ids: list[str],
    ) -> dict[Any, str | None]:
        assert self.use_openai_batches, (
            "_generate_batch_llm_response_openai_from_batch_ids called but "
            "use_openai_batches is False"
        )
        assert isinstance(batch_ids, list) and len(batch_ids) > 0

        keys = list(key_to_messages.keys())
        total_items = len(keys)
        batch_size = self.openai_batch_size
        total_batches = (total_items + batch_size - 1) // batch_size
        assert len(batch_ids) == total_batches, (
            f"Expected {total_batches} batch ids for {total_items} items "
            f"(batch_size={batch_size}), got {len(batch_ids)}"
        )

        model_id = self._get_model_id(self.llm_model_name)
        poll_seconds = self.api_batch_poll_seconds
        assert isinstance(poll_seconds, int) and poll_seconds >= 1

        client = OpenAI(api_key=self.api_key) if self.api_key else OpenAI()

        submitted: list[tuple[str, int, int]] = []
        start_idx = 0
        for batch_id in batch_ids:
            end_idx = min(start_idx + batch_size, total_items)
            submitted.append((batch_id, start_idx, end_idx))
            start_idx = end_idx

        return self._poll_openai_batches(
            client=client,
            submitted=submitted,
            total_items=total_items,
            keys=keys,
            model_id=model_id,
            poll_seconds=poll_seconds,
        )

    def _generate_batch_llm_response_openai(
        self,
        key_to_messages: dict[Any, list[Message]],
    ) -> dict[Any, str | None]:
        """Submit requests via OpenAI Batch API and poll until completion.

        Uses OpenAI's Batch API for ~50% cost savings. This is a synchronous method
        that uploads JSONL files, creates batches, and polls until completion.
        Large request sets are split into chunks of `openai_batch_size`.
        """
        assert self.use_openai_batches, (
            "_generate_batch_llm_response_openai called but use_openai_batches is False"
        )
        model_id = self._get_model_id(self.llm_model_name)
        poll_seconds = self.api_batch_poll_seconds
        batch_size = self.openai_batch_size
        assert isinstance(poll_seconds, int) and poll_seconds >= 1
        assert isinstance(batch_size, int) and batch_size >= 1

        if (
            model_id.startswith("o3")
            or model_id.startswith("o4")
            or model_id.startswith("gpt-5")
        ):
            self.temperature = 1

        keys = list(key_to_messages.keys())
        total_items = len(keys)
        total_batches = (total_items + batch_size - 1) // batch_size
        print(
            f"[APILLMBase] OpenAI batch: {total_items} requests in {total_batches} batch(es), "
            f"polling every {poll_seconds}s"
        )

        client = OpenAI(api_key=self.api_key) if self.api_key else OpenAI()
        # Submit all batches first
        submitted: list[tuple[str, int, int]] = []  # (batch_id, start_idx, end_idx)
        for batch_idx, start_idx in enumerate(
            range(0, total_items, batch_size), start=1
        ):
            end_idx = min(start_idx + batch_size, total_items)
            batch_keys = keys[start_idx:end_idx]

            # Build JSONL requests for this chunk
            requests_list: list[dict[str, Any]] = []
            for i, key in enumerate(batch_keys):
                messages = key_to_messages[key]
                # Convert Message objects to dicts
                messages_dicts = [
                    {
                        "role": m.role.value
                        if hasattr(m.role, "value")
                        else str(m.role),
                        "content": m.content,
                    }
                    for m in messages
                ]

                custom_id = f"req_{start_idx + i}"
                body: dict[str, Any] = {
                    "model": model_id,
                    "messages": messages_dicts,
                    "max_completion_tokens": self.max_tokens,
                    "temperature": float(self.temperature),
                }
                requests_list.append(
                    {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": body,
                    }
                )

            print(
                f"[APILLMBase] Submitting batch {batch_idx}/{total_batches}: "
                f"{len(requests_list)} requests (idx {start_idx}-{end_idx - 1})"
            )

            # Write JSONL file and upload
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False
            ) as f:
                for req in requests_list:
                    json.dump(req, f)
                    f.write("\n")
                input_path = f.name

            try:
                with open(input_path, "rb") as f:
                    input_file = client.files.create(file=f, purpose="batch")

                batch = client.batches.create(
                    input_file_id=input_file.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={"description": "APILLMBase"},
                )
            finally:
                os.unlink(input_path)

            batch_id = batch.id
            assert isinstance(batch_id, str) and len(batch_id) > 0
            print(f"[APILLMBase] OPENAI_BATCH_ID={batch_id}")
            submitted.append((batch_id, start_idx, end_idx))

        return self._poll_openai_batches(
            client=client,
            submitted=submitted,
            total_items=total_items,
            keys=keys,
            model_id=model_id,
            poll_seconds=poll_seconds,
        )

    async def _generate_batch_llm_response(
        self,
        key_to_messages: dict[Any, list[Message]],
    ) -> dict[Any, str | None]:
        assert isinstance(key_to_messages, dict) and len(key_to_messages) > 0

        # Dispatch to OpenAI batch API if enabled
        if self.use_openai_batches:
            return self._generate_batch_llm_response_openai(key_to_messages)

        keys = list(key_to_messages.keys())
        total_items = len(keys)
        print(
            f"[LLM] Starting batched generation for {total_items} items "
            f"(max_concurrent={self.max_concurrent}, retries_per_item={self.max_retries_per_item})"
        )

        # Build requests (processor can take raw requests directly)
        requests: list[ChatCompletionRequest] = []
        for key in keys:
            requests.append(
                ChatCompletionRequest(
                    model=self.llm_model_name,
                    messages=key_to_messages[key],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
            )

        # Configure batch processing
        config = BatchConfig(
            max_concurrent_requests=int(self.max_concurrent),
            max_retries_per_item=self.max_retries_per_item,
            show_progress=True,
            progress_desc="Generating batch of responses",
            print_prompts=False,
            print_responses=False,
            verbose_exceptions=False,
        )

        # Run batch using limiter; no need to specify provider explicitly
        async with ChatLimiter.for_model(
            self.llm_model_name,
            api_key=self.api_key,
            timeout=self.chat_timeout,
            provider=self.provider,
        ) as limiter:
            results = await process_chat_completion_batch(limiter, requests, config)

        # Map results back to input keys
        key_to_response: dict[Any, str | None] = {}
        for i, result in enumerate(results):
            if result.success and result.result and result.result.choices:
                key_to_response[keys[i]] = result.result.choices[0].message.content
            else:
                key_to_response[keys[i]] = None

        assert len(key_to_response) == len(keys)
        return key_to_response

    async def _generate_single_llm_response(
        self,
        messages: list[Message],
    ) -> str | None:
        max_retries = self.max_retries_per_item
        response: ChatCompletionResponse | None = None
        for attempt in range(max_retries):
            response = await self.limiter.chat_completion(
                model=self.llm_model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            if response.success:
                break  # Success, exit retry loop
            else:
                print(f"Request failed: {response}")
                await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff

        if response is None or response.choices is None or len(response.choices) == 0:
            print(
                f"There was an error generating a response for the following messages: {messages}"
            )
            print(f"Response: {response}")
            return None
        return response.choices[0].message.content
