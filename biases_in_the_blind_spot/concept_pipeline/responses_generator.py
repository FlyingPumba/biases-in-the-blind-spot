import asyncio
import concurrent.futures
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request as AnthropicRequest
from chat_limiter import (
    BatchConfig,
    ChatLimiter,
    create_chat_completion_requests,
    process_chat_completion_batch,
)
from openai import OpenAI

from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_dataset import (
    ConceptPipelineDataset,
)
from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_result import (
    ConceptPipelineResult,
)


@dataclass
class ResponsesGenerator:
    model_name: str = "google/gemma-3-12b-it"
    temperature: float = 0.7
    max_tokens: int = 6000
    batch_size: int = 20000
    max_concurrent_requests: int = 200
    max_retries_per_item: int = 10
    request_timeout: float = 600.0
    # Poll interval (seconds) for provider batch APIs (OpenAI/Anthropic).
    # Default is 3 minutes.
    api_batch_poll_seconds: int = 180
    # If True, use the provider's batch API (synchronous polling).
    # Only allowed when model_name starts with the corresponding provider prefix.
    use_openai_batches: bool = False
    use_anthropic_batches: bool = False
    seed = 42
    top_p = 0.95

    @property
    def config(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "batch_size": self.batch_size,
            "use_openai_batches": self.use_openai_batches,
            "use_anthropic_batches": self.use_anthropic_batches,
            "api_batch_poll_seconds": self.api_batch_poll_seconds,
            "seed": self.seed,
            "top_p": self.top_p,
            "max_concurrent_requests": self.max_concurrent_requests,
            "max_retries_per_item": self.max_retries_per_item,
            "request_timeout": self.request_timeout,
        }

    def generate(
        self,
        input_template: str,
        input_parameters_list: list[dict[str, str]],
        temperature: float | None = None,
        max_new_tokens: int | None = None,
        batch_size: int | None = None,
    ) -> list[str]:
        """Generate responses using remote APIs (OpenRouter, OpenAI, or Anthropic)."""
        if temperature is None:
            temperature = self.temperature
        if max_new_tokens is None:
            max_new_tokens = self.max_tokens
        if batch_size is None:
            batch_size = self.batch_size
        assert isinstance(batch_size, int) and batch_size >= 1
        assert not (self.use_openai_batches and self.use_anthropic_batches), (
            "Only one of use_openai_batches/use_anthropic_batches can be True"
        )
        total_items = len(input_parameters_list)
        total_batches = (total_items + batch_size - 1) // batch_size
        print(
            f"[ResponsesGenerator] Preparing to generate {total_items} responses "
            f"(batch_size={batch_size}, total_batches={total_batches})"
        )

        if self.use_openai_batches:
            start_time = time.time()
            print(
                f"[ResponsesGenerator] Starting OpenAI batched generation: "
                f"{total_items} inputs, {total_batches} batches"
            )
            resps = self._generate_openai_batched(
                input_template,
                input_parameters_list,
                temperature,
                max_new_tokens,
                batch_size,
            )
            end_time = time.time()
            print(
                f"[ResponsesGenerator] OpenAI batched generation completed in "
                f"{end_time - start_time:.2f} seconds"
            )
        elif self.use_anthropic_batches:
            start_time = time.time()
            print(
                f"[ResponsesGenerator] Starting Anthropic batched generation: "
                f"{total_items} inputs, {total_batches} batches"
            )
            resps = self._generate_anthropic_batched(
                input_template,
                input_parameters_list,
                temperature,
                max_new_tokens,
                batch_size,
            )
            end_time = time.time()
            print(
                f"[ResponsesGenerator] Anthropic batched generation completed in "
                f"{end_time - start_time:.2f} seconds"
            )
        else:
            start_time = time.time()
            print(
                f"[ResponsesGenerator] Starting OpenRouter generation: "
                f"{total_items} inputs, {total_batches} batches"
            )
            resps = self._generate_open_router(
                input_template,
                input_parameters_list,
                temperature,
                max_new_tokens,
                batch_size,
            )
            end_time = time.time()
            print(
                f"[ResponsesGenerator] OpenRouter generation completed in "
                f"{end_time - start_time:.2f} seconds"
            )
        return resps

    @staticmethod
    def _strip_provider_prefix(model_name: str, provider: str) -> str:
        assert isinstance(model_name, str) and len(model_name) > 0
        assert provider in {"openai", "anthropic"}
        if model_name.startswith(f"{provider}/"):
            model_id = model_name.split("/", 1)[1]
        elif model_name.startswith(f"{provider}:"):
            model_id = model_name.split(":", 1)[1]
        else:
            raise AssertionError(
                f"model_name must start with '{provider}/' or '{provider}:' when using {provider} batch mode; got {model_name!r}"
            )
        assert isinstance(model_id, str) and len(model_id) > 0
        return model_id

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

    def _generate_openai_batched(
        self,
        input_template: str,
        input_parameters_list: list[dict[str, str]],
        temperature: float,
        max_new_tokens: int,
        batch_size: int,
    ) -> list[str]:
        """Submit OpenAI chat completion requests via the OpenAI Batch API and poll until completion.

        Strategy:
        - Submit all batches first (one provider batch per chunk of size `batch_size`).
        - Then poll all pending batches (every `api_batch_poll_seconds`) until completion.
        - Stream results into a preallocated `responses` list keyed by `custom_id == "req_{global_idx}"`.
        """
        assert self.use_openai_batches, (
            "_generate_openai_batched called but use_openai_batches is False"
        )
        model_id = self._strip_provider_prefix(self.model_name, "openai")
        assert isinstance(max_new_tokens, int) and max_new_tokens >= 1
        assert isinstance(batch_size, int) and batch_size >= 1

        client = OpenAI()
        poll_seconds = self.api_batch_poll_seconds
        assert isinstance(poll_seconds, int) and poll_seconds >= 1
        total_items = len(input_parameters_list)
        total_batches = (total_items + batch_size - 1) // batch_size
        responses: list[str] = [""] * total_items

        submitted: list[tuple[str, int, int]] = []
        for batch_idx, start_idx in enumerate(
            range(0, total_items, batch_size), start=1
        ):
            end_idx = min(start_idx + batch_size, total_items)
            batch_params = input_parameters_list[start_idx:end_idx]
            try:
                prompts = [input_template.format(**p) for p in batch_params]
            except Exception as e:
                print(f"Error formatting prompts: {e}")
                print(f"Params: {batch_params[0]}")
                print(f"Task prompt template: {input_template}")
                raise

            requests: list[dict[str, Any]] = []
            for i, prompt in enumerate(prompts):
                custom_id = f"req_{start_idx + i}"
                body: dict[str, Any] = {
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if model_id.startswith(("o1", "o3", "o4")):
                    body["max_completion_tokens"] = max_new_tokens
                    body["temperature"] = 1.0
                else:
                    body["max_tokens"] = max_new_tokens
                    body["temperature"] = float(temperature)
                    body["top_p"] = float(self.top_p)

                requests.append(
                    {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": body,
                    }
                )

            print(
                f"[ResponsesGenerator] OpenAI batch {batch_idx}/{total_batches}: "
                f"{len(requests)} requests (start={start_idx}, end={end_idx})"
            )

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False
            ) as f:
                for req in requests:
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
                    metadata={"description": "ResponsesGenerator"},
                )
            finally:
                os.unlink(input_path)

            batch_id = batch.id
            assert isinstance(batch_id, str) and len(batch_id) > 0
            print(f"[ResponsesGenerator] OPENAI_BATCH_ID={batch_id}")
            submitted.append((batch_id, start_idx, end_idx))

        print(
            f"[ResponsesGenerator] Submitted {len(submitted)} OpenAI batches. "
            f"Polling every {poll_seconds}s until all complete..."
        )

        pending: dict[str, tuple[int, int]] = {bid: (s, e) for bid, s, e in submitted}
        while len(pending) > 0:
            status_counts: dict[str, int] = {}
            completed_now: list[str] = []
            for batch_id, (start_idx, end_idx) in list(pending.items()):
                try:
                    batch = client.batches.retrieve(batch_id)
                except Exception as e:
                    if self._is_transient_batch_poll_error(e):
                        print(
                            f"[ResponsesGenerator] OpenAI transient error retrieving batch "
                            f"(treating as in_progress): batch_id={batch_id}, error={type(e).__name__}: {e}"
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

                try:
                    if batch.status != "completed":
                        raise RuntimeError(
                            f"OpenAI batch did not complete successfully: batch_id={batch_id}, status={batch.status}"
                        )

                    # Handle case where output_file_id is not yet populated (race condition)
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
                                f"\n[ResponsesGenerator] ERROR: OpenAI batch {batch_id} failed - "
                                f"all {getattr(req_counts, 'total', '?')} requests failed."
                            )
                            print(f"[ResponsesGenerator] request_counts={req_counts}")
                            first_error_msg = None
                            if batch.error_file_id:
                                try:
                                    err_content = client.files.content(
                                        batch.error_file_id
                                    )
                                    print(
                                        f"[ResponsesGenerator] Errors from batch {batch_id}:"
                                    )
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
                                            print(
                                                f"  (failed to parse error line): {line}"
                                            )
                                except Exception as e:
                                    print(
                                        f"  (could not read error file {batch.error_file_id}: {e})"
                                    )
                            raise RuntimeError(
                                f"OpenAI batch failed: all {getattr(req_counts, 'total', '?')} "
                                f"requests failed. batch_id={batch_id}. "
                                f"First error: {first_error_msg or 'unknown'}"
                            )
                        # output_file_id not ready yet, will retry on next poll
                        print(
                            f"[ResponsesGenerator] Batch {batch_id} completed but output_file_id "
                            f"not ready, will retry on next poll. request_counts={req_counts}"
                        )
                        continue

                    file_response = client.files.content(batch.output_file_id)
                    for line in file_response.text.splitlines():
                        obj = json.loads(line)
                        custom_id = obj.get("custom_id")
                        assert isinstance(custom_id, str) and len(custom_id) > 0

                        if obj.get("error") is not None:
                            print(
                                f"[ResponsesGenerator] Request {custom_id} failed in batch "
                                f"{batch_id}: {obj['error']}"
                            )
                            continue

                        body = obj.get("response", {}).get("body", {})
                        choices = body.get("choices", [])
                        if not (
                            choices and choices[0].get("message", {}).get("content")
                        ):
                            print(
                                f"[ResponsesGenerator] Request {custom_id} missing message.content "
                                f"in batch {batch_id}: body_keys={list(body.keys())}"
                            )
                            continue

                        assert custom_id.startswith("req_")
                        idx_str = custom_id.split("_", 1)[1]
                        idx = int(idx_str)
                        assert 0 <= idx < total_items
                        responses[idx] = choices[0]["message"]["content"]

                    # Process error file for failed requests (partial failures)
                    if batch.error_file_id is not None:
                        err_response = client.files.content(batch.error_file_id)
                        print(f"[ResponsesGenerator] Errors from batch {batch_id}:")
                        for line in err_response.text.splitlines():
                            try:
                                err_obj = json.loads(line)
                                custom_id = err_obj.get("custom_id", "?")
                                error_info = err_obj.get("error") or err_obj.get(
                                    "response", {}
                                ).get("body", {}).get("error", {})
                                error_msg = (
                                    error_info.get("message", str(error_info))
                                    if isinstance(error_info, dict)
                                    else str(error_info)
                                )
                                error_type = (
                                    error_info.get("type", "unknown")
                                    if isinstance(error_info, dict)
                                    else "unknown"
                                )
                                print(f"  {custom_id}: [{error_type}] {error_msg}")
                                # Store empty string for failed requests
                                if custom_id.startswith("req_"):
                                    idx = int(custom_id.split("_", 1)[1])
                                    if idx < total_items and responses[idx] == "":
                                        responses[idx] = (
                                            ""  # Already empty, but explicit
                                        )
                            except json.JSONDecodeError:
                                print(f"  (failed to parse error line): {line}")
                except Exception as e:
                    if self._is_transient_batch_poll_error(e):
                        print(
                            f"[ResponsesGenerator] OpenAI transient error processing completed batch "
                            f"(treating as in_progress): batch_id={batch_id}, error={type(e).__name__}: {e}"
                        )
                        continue
                    raise

                # Sanity: at least attempt to fill the range (errors may leave empty strings).
                assert 0 <= start_idx <= end_idx <= total_items
                del pending[batch_id]
                completed_now.append(batch_id)

            print(
                f"[ResponsesGenerator] OpenAI batches pending={len(pending)} "
                f"(just_completed={len(completed_now)}): {status_counts}"
            )
            if len(pending) > 0:
                time.sleep(poll_seconds)

        # Check for failed requests (empty strings) and fallback to non-batched generation
        failed_indices = [i for i, r in enumerate(responses) if r == ""]
        if len(failed_indices) == total_items:
            # ALL requests failed - raise error (likely a config issue)
            raise RuntimeError(
                f"[ResponsesGenerator] OpenAI batch failed: all {total_items} requests failed. "
                f"See error messages above for details."
            )
        elif len(failed_indices) > 0:
            # SOME requests failed - fallback to non-batched generation
            print(
                f"[ResponsesGenerator] {len(failed_indices)} of {total_items} requests failed in batches. "
                f"Falling back to non-batched generation for failed requests..."
            )
            failed_params = [input_parameters_list[i] for i in failed_indices]
            # Temporarily disable batching for fallback
            prev_use_openai_batches = self.use_openai_batches
            prev_use_anthropic_batches = self.use_anthropic_batches
            self.use_openai_batches = False
            self.use_anthropic_batches = False
            try:
                fallback_responses = self.generate(
                    input_template,
                    failed_params,
                    temperature,
                    max_new_tokens,
                    batch_size,
                )
            finally:
                self.use_openai_batches = prev_use_openai_batches
                self.use_anthropic_batches = prev_use_anthropic_batches
            for i, idx in enumerate(failed_indices):
                responses[idx] = fallback_responses[i]
            print(
                f"[ResponsesGenerator] Fallback generation completed for {len(failed_indices)} requests."
            )

        assert len(responses) == len(input_parameters_list)
        return responses

    def _generate_anthropic_batched(
        self,
        input_template: str,
        input_parameters_list: list[dict[str, str]],
        temperature: float,
        max_new_tokens: int,
        batch_size: int,
    ) -> list[str]:
        """Submit Anthropic message requests via the Anthropic Batch API and poll until completion.

        Strategy:
        - Submit all batches first (one provider batch per chunk of size `batch_size`).
        - Then poll all pending batches (every `api_batch_poll_seconds`) until they end.
        - Stream results into a preallocated `responses` list keyed by `custom_id == "req_{global_idx}"`.
        """
        assert self.use_anthropic_batches, (
            "_generate_anthropic_batched called but use_anthropic_batches is False"
        )
        model_id = self._strip_provider_prefix(self.model_name, "anthropic")
        assert isinstance(max_new_tokens, int) and max_new_tokens >= 1
        assert isinstance(batch_size, int) and batch_size >= 1

        client = Anthropic()
        poll_seconds = self.api_batch_poll_seconds
        assert isinstance(poll_seconds, int) and poll_seconds >= 1
        total_items = len(input_parameters_list)
        total_batches = (total_items + batch_size - 1) // batch_size
        responses: list[str] = [""] * total_items

        submitted: list[tuple[str, int, int]] = []
        for batch_idx, start_idx in enumerate(
            range(0, total_items, batch_size), start=1
        ):
            end_idx = min(start_idx + batch_size, total_items)
            batch_params = input_parameters_list[start_idx:end_idx]
            try:
                prompts = [input_template.format(**p) for p in batch_params]
            except Exception as e:
                print(f"Error formatting prompts: {e}")
                print(f"Params: {batch_params[0]}")
                print(f"Task prompt template: {input_template}")
                raise

            requests: list[AnthropicRequest] = []
            for i, prompt in enumerate(prompts):
                custom_id = f"req_{start_idx + i}"
                params = MessageCreateParamsNonStreaming(
                    model=model_id,
                    max_tokens=max_new_tokens,
                    temperature=float(temperature),
                    top_p=float(self.top_p),
                    messages=[{"role": "user", "content": prompt}],
                )
                requests.append(AnthropicRequest(custom_id=custom_id, params=params))

            print(
                f"[ResponsesGenerator] Anthropic batch {batch_idx}/{total_batches}: "
                f"{len(requests)} requests (start={start_idx}, end={end_idx})"
            )
            message_batch = client.messages.batches.create(requests=requests)
            batch_id = message_batch.id
            assert isinstance(batch_id, str) and len(batch_id) > 0
            print(f"[ResponsesGenerator] ANTHROPIC_BATCH_ID={batch_id}")
            submitted.append((batch_id, start_idx, end_idx))

        print(
            f"[ResponsesGenerator] Submitted {len(submitted)} Anthropic batches. "
            f"Polling every {poll_seconds}s until all end..."
        )

        pending: dict[str, tuple[int, int]] = {bid: (s, e) for bid, s, e in submitted}
        while len(pending) > 0:
            status_counts: dict[str, int] = {}
            completed_now: list[str] = []
            for batch_id, (start_idx, end_idx) in list(pending.items()):
                try:
                    message_batch = client.messages.batches.retrieve(batch_id)
                except Exception as e:
                    if self._is_transient_batch_poll_error(e):
                        print(
                            f"[ResponsesGenerator] Anthropic transient error retrieving batch "
                            f"(treating as in_progress): batch_id={batch_id}, error={type(e).__name__}: {e}"
                        )
                        status_counts["in_progress"] = (
                            status_counts.get("in_progress", 0) + 1
                        )
                        continue
                    raise
                status = str(message_batch.processing_status)
                status_counts[status] = status_counts.get(status, 0) + 1
                if status == "ended":
                    pass
                elif status in {"in_progress"}:
                    continue
                else:
                    raise RuntimeError(
                        f"Anthropic batch ended in unexpected state: batch_id={batch_id}, processing_status={message_batch.processing_status}"
                    )

                try:
                    for result in client.messages.batches.results(batch_id):
                        custom_id = result.custom_id
                        assert isinstance(custom_id, str) and len(custom_id) > 0
                        r = result.result
                        match r.type:
                            case "succeeded":
                                message = r.message
                                content = message.content
                                assert content is not None and len(content) > 0
                                if len(content) == 1 and content[0].type == "text":
                                    text = content[0].text
                                elif (
                                    len(content) == 2
                                    and content[0].type == "thinking"
                                    and content[1].type == "text"
                                ):
                                    text = f"<think>{content[0].thinking}</think>{content[1].text}"
                                else:
                                    raise AssertionError(
                                        f"Unexpected Anthropic content blocks for {custom_id}: {[c.type for c in content]}"
                                    )

                                assert custom_id.startswith("req_")
                                idx_str = custom_id.split("_", 1)[1]
                                idx = int(idx_str)
                                assert 0 <= idx < total_items
                                responses[idx] = text
                            case "errored":
                                raise RuntimeError(
                                    f"Anthropic batch request errored: batch_id={batch_id}, custom_id={custom_id}, error={r.error}"
                                )
                            case "expired":
                                raise RuntimeError(
                                    f"Anthropic batch request expired: batch_id={batch_id}, custom_id={custom_id}"
                                )
                            case _:
                                raise AssertionError(
                                    f"Unexpected Anthropic batch result type: {r.type}"
                                )
                except Exception as e:
                    if self._is_transient_batch_poll_error(e):
                        print(
                            f"[ResponsesGenerator] Anthropic transient error fetching batch results "
                            f"(treating as in_progress): batch_id={batch_id}, error={type(e).__name__}: {e}"
                        )
                        continue
                    raise

                assert 0 <= start_idx <= end_idx <= total_items
                del pending[batch_id]
                completed_now.append(batch_id)

            print(
                f"[ResponsesGenerator] Anthropic batches pending={len(pending)} "
                f"(just_ended={len(completed_now)}): {status_counts}"
            )
            if len(pending) > 0:
                time.sleep(poll_seconds)

        assert len(responses) == len(input_parameters_list)
        return responses

    def _generate_open_router(
        self,
        input_template: str,
        input_parameters_list: list[dict[str, str]],
        temperature: float,
        max_new_tokens: int,
        batch_size: int,
    ) -> list[str]:
        def _format_prompts_batch(params: list[dict[str, str]]) -> list[str]:
            try:
                prompts = [input_template.format(**p) for p in params]
            except Exception as e:
                print(f"Error formatting prompts: {e}")
                print(f"Params: {params[0]}")
                print(f"Task prompt template: {input_template}")
                raise e
            return prompts

        async def _generate_remote_batch(prompts: list[str]) -> list[str]:
            config = BatchConfig(
                max_concurrent_requests=self.max_concurrent_requests,
                max_retries_per_item=self.max_retries_per_item,
                print_prompts=False,
                print_responses=False,
                verbose_exceptions=True,
                verbose_timeouts=True,
            )
            requests = create_chat_completion_requests(
                model=self.model_name,
                prompts=prompts,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=self.top_p,
                seed=self.seed,
                providers=["hyperbolic/bf16"] if "qwq" in self.model_name else None,
            )

            async with ChatLimiter.for_model(
                self.model_name,
                timeout=self.request_timeout,
                provider="openrouter",
                base_backoff=1.0,
            ) as limiter:
                print(
                    f"[ResponsesGenerator] Dispatching {len(prompts)} prompts to OpenRouter "
                    f"(max_concurrent={config.max_concurrent_requests}, "
                    f"retries_per_item={config.max_retries_per_item})"
                )
                results = await process_chat_completion_batch(limiter, requests, config)

            responses: list[str] = []
            for i, result in enumerate(results):
                if result.success and result.result:
                    response = result.result
                    if hasattr(response, "choices") and response.choices:
                        choice = response.choices[0]
                        content = choice.message.content
                        responses.append(content)
                    else:
                        responses.append(str(response))
                else:
                    if not result.success:
                        msg = str(result.error_message or "")
                        attempts = getattr(result, "attempt_count", None)
                        detail_parts = [msg]
                        if attempts is not None:
                            detail_parts.append(f"attempts={attempts}")
                        last_err = getattr(result.item, "last_error", None)
                        if last_err is not None:
                            detail_parts.append(f"last_error={repr(last_err)}")
                        if result.status_code is not None:
                            detail_parts.append(f"status={result.status_code}")
                        if result.response_headers:
                            detail_parts.append(f"headers={result.response_headers}")
                        print(f"Batch request {i} failed: " + " | ".join(detail_parts))
                        lower_msg = msg.lower()
                        if (
                            "insufficient credits" in lower_msg
                            or "openrouter.ai/settings/credits" in lower_msg
                        ):
                            raise RuntimeError(f"OpenRouter credits exhausted: {msg}")
                    else:
                        print(f"Batch request {i} succeeded but response is empty.")
                    responses.append("")

            assert len(responses) == len(prompts)
            return responses

        all_remote_responses: list[str] = []
        total_items = len(input_parameters_list)
        total_batches = (total_items + batch_size - 1) // batch_size
        for batch_idx, start_idx in enumerate(
            range(0, total_items, batch_size), start=1
        ):
            end_idx = min(start_idx + batch_size, len(input_parameters_list))
            batch_params = input_parameters_list[start_idx:end_idx]
            batch_prompts = _format_prompts_batch(batch_params)
            print(
                f"[ResponsesGenerator] OpenRouter batch {batch_idx}/{total_batches}: "
                f"{len(batch_prompts)} prompts (start={start_idx}, end={end_idx})"
            )

            try:
                asyncio.get_running_loop()

                def _runner(prompts: list[str] = batch_prompts) -> list[str]:
                    return asyncio.run(_generate_remote_batch(prompts))

                with concurrent.futures.ThreadPoolExecutor() as ex:
                    fut = ex.submit(_runner)
                    responses = fut.result()
            except RuntimeError:
                responses = asyncio.run(_generate_remote_batch(batch_prompts))

            assert len(responses) == (end_idx - start_idx)
            all_remote_responses.extend(responses)

        assert len(all_remote_responses) == len(input_parameters_list)
        return all_remote_responses

    def export_baseline_responses_html(
        self,
        dataset: ConceptPipelineDataset,
        result: ConceptPipelineResult,
        output_dir: Path,
    ) -> None:
        """Export baseline responses as a single HTML file at the figures root.

        File name: baseline_responses.html
        """
        figures_root = result.get_figures_root(output_dir)
        input_template = dataset.input_template
        input_parameters = dataset.input_parameters
        varying_input_param_name = dataset.varying_input_param_name
        parsed_labels_mapping = result.parsed_labels_mapping
        baseline_responses_by_input = result.baseline_responses_by_input
        baseline_acceptances_by_input = result.baseline_acceptances_by_input

        assert isinstance(parsed_labels_mapping, dict)
        assert 1 in parsed_labels_mapping and 0 in parsed_labels_mapping, (
            "Parsed labels mapping should contain 1 and 0"
        )
        assert (
            isinstance(baseline_responses_by_input, dict)
            and len(baseline_responses_by_input) > 0
        ), "Baseline responses by input are not available"
        assert (
            isinstance(baseline_acceptances_by_input, dict)
            and len(baseline_acceptances_by_input) > 0
        ), "Baseline acceptances by input are not available"

        # Collect fixed parameters (all except the varying one)
        fixed_params = {
            k: v for k, v in input_parameters.items() if k != varying_input_param_name
        }

        parts: list[str] = []
        # Input template section
        parts.append("<h1>Input template</h1>")
        parts.append(
            '<pre style="white-space: pre-wrap; word-break: break-word; overflow-wrap: anywhere;">'
            + input_template
            + "</pre>"
        )
        # Fixed parameters section (non-varying)
        parts.append("<h1>Fixed input parameters</h1>")
        if len(fixed_params) > 0:
            for key in sorted(fixed_params.keys()):
                parts.append(f"<h2>{key}</h2>")
                val = fixed_params[key]
                assert isinstance(val, str)
                parts.append(
                    '<pre style="white-space: pre-wrap; word-break: break-word; overflow-wrap: anywhere;">'
                    + val
                    + "</pre>"
                )
        else:
            parts.append("<p>(No fixed parameters)</p>")
        # Inputs under analysis (sanitized)
        parts.append("<h1>Sanitized inputs under analysis</h1>")
        assert isinstance(dataset.sanitized_varying_inputs, dict)
        for input_idx in sorted(dataset.sanitized_varying_inputs.keys()):
            parts.append(f"<h2>Input {input_idx}</h2>")
            input_text = dataset.sanitized_varying_inputs[input_idx]
            parts.append(
                '<pre style="white-space: pre-wrap; word-break: break-word; overflow-wrap: anywhere;">'
                + input_text
                + "</pre>"
            )

        for input_idx in sorted(baseline_responses_by_input.keys()):
            parts.append(f"<h2>Input {input_idx}</h2>")
            resp_map = baseline_responses_by_input[input_idx]
            assert isinstance(resp_map, dict) and len(resp_map) > 0
            for response_idx in sorted(resp_map.keys()):
                resp = resp_map[response_idx]
                parsed_resp = baseline_acceptances_by_input[input_idx][response_idx]
                label = (
                    parsed_labels_mapping[1]
                    if parsed_resp == 1
                    else (parsed_labels_mapping[0] if parsed_resp == 0 else "UNKNOWN")
                )
                parts.append(f"<h3>Response {response_idx} ({label})</h3>")
                parts.append(
                    '<pre style="white-space: pre-wrap; word-break: break-word; overflow-wrap: anywhere;">'
                    + resp
                    + "</pre>"
                )

        html = "\n\n".join(parts)
        out_path = os.path.join(figures_root, "baseline_responses.html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
