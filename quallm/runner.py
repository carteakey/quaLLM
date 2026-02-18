"""Send prompts to models and collect results + performance metrics."""

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import requests

from .config import ModelConfig, PromptConfig

log = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Result of running a single prompt against a single model."""

    model_name: str
    prompt_name: str
    raw_output: str
    code_output: Optional[str] = None

    # Token counts
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # Performance (tokens / second)
    prompt_tk_s: float = 0.0
    gen_tk_s: float = 0.0

    # Wall-clock
    total_time_s: float = 0.0

    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def _extract_code_blocks(text: str) -> Optional[str]:
    """Extract fenced code blocks (```python ... ``` or ``` ... ```) from text."""
    # Try python-specific blocks first
    pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return "\n\n".join(m.strip() for m in matches)
    return None


def run_prompt(
    model: ModelConfig,
    prompt: PromptConfig,
    max_tokens: int = 16384,
    timeout: float = 600,
) -> RunResult:
    """Send a prompt to the model's OpenAI-compatible API and collect results.

    Uses /v1/chat/completions (non-streaming) and reads back the usage +
    timings fields that llama-server provides.
    """
    url = f"{model.base_url}/v1/chat/completions"

    messages = []
    if prompt.system_prompt:
        messages.append({"role": "system", "content": prompt.system_prompt})
    messages.append({"role": "user", "content": prompt.user_prompt})

    payload = {
        "model": model.alias,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {model.api_key}",
    }

    log.info("Sending prompt '%s' to %s …", prompt.name, model.name)

    t0 = time.monotonic()
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    wall_time = time.monotonic() - t0
    resp.raise_for_status()

    data = resp.json()

    # Extract content
    raw_output = data["choices"][0]["message"]["content"]

    # Extract code if requested
    code_output = None
    if prompt.extract_code:
        code_output = _extract_code_blocks(raw_output)

    # Usage stats
    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    # llama.cpp timings (may be in usage or top-level)
    timings = data.get("timings", usage.get("timings", {}))
    prompt_tk_s = timings.get("prompt_per_second", 0.0)
    gen_tk_s = timings.get("predicted_per_second", 0.0)

    # Fallback: compute from wall time if timings unavailable
    if gen_tk_s == 0.0 and completion_tokens > 0 and wall_time > 0:
        gen_tk_s = completion_tokens / wall_time

    result = RunResult(
        model_name=model.name,
        prompt_name=prompt.name,
        raw_output=raw_output,
        code_output=code_output,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_tk_s=round(prompt_tk_s, 2),
        gen_tk_s=round(gen_tk_s, 2),
        total_time_s=round(wall_time, 2),
    )

    log.info(
        "Done: %s → %d tokens in %.1fs (%.1f tk/s gen, %.1f tk/s prompt)",
        model.name,
        completion_tokens,
        wall_time,
        gen_tk_s,
        prompt_tk_s,
    )

    return result
