import os
import time

import pytest

from biases_in_the_blind_spot.concept_pipeline.responses_generator import (
    ResponsesGenerator,
)

# Always skipped for now (live API integration tests). Remove this marker to re-enable.
pytestmark = pytest.mark.skip(
    reason="Live provider batch tests are disabled by default (remove this marker to run)."
)


def test_generate_openai_batched_live() -> None:
    if os.getenv("OPENAI_API_KEY", "") == "":
        pytest.skip("OPENAI_API_KEY is not set.")

    print(f"[live-test][openai] starting at {time.time():.0f}", flush=True)
    gen = ResponsesGenerator(
        model_name="openai/gpt-4o-mini",
        use_openai_batches=True,
        use_anthropic_batches=False,
        api_batch_poll_seconds=30,
        temperature=0.0,
        max_tokens=64,
        batch_size=1,
    )

    template = "Reply with exactly the single word 'OK' and nothing else. Prompt: {x}"
    params = [{"x": "test_0"}, {"x": "test_1"}]
    print(f"[live-test][openai] calling generate (n={len(params)})", flush=True)
    out = gen.generate(template, params)
    print("[live-test][openai] generate returned", flush=True)
    assert isinstance(out, list) and len(out) == len(params)
    for s in out:
        assert isinstance(s, str)
        assert "OK" in s.strip().upper()


def test_generate_anthropic_batched_live() -> None:
    if os.getenv("ANTHROPIC_API_KEY", "") == "":
        pytest.skip("ANTHROPIC_API_KEY is not set.")

    print(f"[live-test][anthropic] starting at {time.time():.0f}", flush=True)
    gen = ResponsesGenerator(
        model_name="anthropic/claude-3-5-haiku-20241022",
        use_openai_batches=False,
        use_anthropic_batches=True,
        api_batch_poll_seconds=30,
        temperature=0.0,
        max_tokens=64,
        batch_size=1,
    )

    template = "Reply with exactly the single word 'OK' and nothing else. Prompt: {x}"
    params = [{"x": "test_0"}, {"x": "test_1"}]
    print(f"[live-test][anthropic] calling generate (n={len(params)})", flush=True)
    out = gen.generate(template, params)
    print("[live-test][anthropic] generate returned", flush=True)
    assert isinstance(out, list) and len(out) == len(params)
    for s in out:
        assert isinstance(s, str)
        assert "OK" in s.strip().upper()
