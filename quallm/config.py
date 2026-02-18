"""Model and prompt configuration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    """Configuration for a model served via llama-server."""

    name: str
    alias: str
    model_path: str
    port: int = 8001
    api_key: str = "dummy"
    server_binary: str = "./vendor/llama.cpp/build/bin/llama-server"
    host: str = "0.0.0.0"

    # Extra llama-server CLI flags (key→value, or key→True for flags)
    server_args: dict = field(default_factory=dict)

    # Environment variables to set before launching
    env: dict = field(default_factory=lambda: {
        "LLAMA_SET_ROWS": "1",
        "GGML_CUDA_GRAPH_OPT": "1",
    })

    # CPU affinity (taskset mask), e.g. "0-11"
    taskset_cpus: Optional[str] = "0-11"

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def safe_name(self) -> str:
        """Filesystem-safe version of the model name."""
        return self.name.lower().replace(" ", "_").replace("/", "_").replace("-", "_")


@dataclass
class PromptConfig:
    """A prompt to send to models."""

    name: str
    user_prompt: str
    system_prompt: str = "You are a helpful assistant. Respond only with the requested code, no explanations."
    eval_type: str = "human"          # "human" or "auto"
    extract_code: bool = True         # try to extract code blocks from response


def load_model_config(path: str | Path) -> ModelConfig:
    """Load a ModelConfig from a YAML file."""
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)
    return ModelConfig(**data)


def load_prompt(path: str | Path) -> PromptConfig:
    """Load a PromptConfig from a .txt or .yaml file.

    If .txt, the file contents become the user_prompt with the filename
    (minus extension) used as the name.
    If .yaml, all PromptConfig fields can be specified.
    """
    path = Path(path)

    if path.suffix in (".yaml", ".yml"):
        with open(path) as f:
            data = yaml.safe_load(f)
        return PromptConfig(**data)

    # Plain text: entire file is the prompt
    text = path.read_text().strip()
    name = path.stem.replace("_", " ").replace("-", " ").title()
    return PromptConfig(name=name, user_prompt=text)
