import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chat_limiter import Message, MessageRole
from redlines import Redlines

from biases_in_the_blind_spot.concept_pipeline.api_llm_base import APILLMBase

DEFAULT_SANITIZE_PROMPT = """You are a precise copy editor. You will receive a single text block.

Task: Correct typos, obvious misspellings, and broken grammar while preserving meaning, tone, formatting, and structure. Do not add or remove content beyond necessary corrections. Do not explain your edits.

Return ONLY the corrected text, with no preface or extra markers.

<text>
{text}
</text>
"""


@dataclass
class InputSanitizer(APILLMBase):
    """Sanitizes selected string fields in an input-parameter mapping via an LLM."""

    llm_model_name: str = "gpt-4o-mini"
    temperature: float = 0.2
    sanitize_prompt: str = DEFAULT_SANITIZE_PROMPT

    @property
    def config(self) -> dict[str, Any]:
        base = super().config
        base.update({})
        return base

    async def sanitize_text_batch(self, texts_by_key: dict[Any, str]) -> dict[Any, str]:
        """Sanitize multiple texts in parallel.

        Args:
            texts_by_key: Mapping from arbitrary keys to the original text to sanitize.

        Returns:
            Mapping from the same keys to sanitized texts.
        """
        assert isinstance(texts_by_key, dict) and len(texts_by_key) > 0
        assert all(isinstance(k, object) for k in texts_by_key.keys())
        assert all(isinstance(v, str) and len(v) > 0 for v in texts_by_key.values())

        # Build batch messages keyed by user-provided keys
        key_to_messages: dict[Any, list[Message]] = {}
        for key, text in texts_by_key.items():
            messages = [
                Message(
                    role=MessageRole.USER,
                    content=self.sanitize_prompt.format(text=text),
                )
            ]
            key_to_messages[key] = messages

        results = await self._generate_batch_llm_response(key_to_messages)

        cleaned_by_key: dict[Any, str] = {}
        for key, response in results.items():
            assert response is not None
            cleaned = response.strip()
            assert isinstance(cleaned, str) and len(cleaned) > 0
            cleaned_by_key[key] = cleaned

        # Ensure coverage preserved
        assert set(cleaned_by_key.keys()) == set(texts_by_key.keys())
        return cleaned_by_key

    def export_sanitization_diff(
        self,
        figures_root_directory: str | Path,
        original_text: str,
        sanitized_text: str,
        varying_input_param_name: str,
    ) -> None:
        """Write an HTML diff for original vs sanitized text under figures_root.

        File name pattern: sanitization_diff__<varying_input_param_name>.html
        """
        assert isinstance(figures_root_directory, str | Path)
        assert isinstance(original_text, str) and len(original_text) > 0
        assert isinstance(sanitized_text, str) and len(sanitized_text) > 0
        assert (
            isinstance(varying_input_param_name, str)
            and len(varying_input_param_name) > 0
        )

        root = str(figures_root_directory)
        os.makedirs(root, exist_ok=True)

        diff = Redlines(original_text, sanitized_text)
        html = diff.output_markdown

        out_path = os.path.join(root, f"sanitized_{varying_input_param_name}.html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
