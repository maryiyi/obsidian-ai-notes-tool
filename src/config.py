"""Configuration and CLI parsing module for obsidian-ai-notes-tool.

Handles environment variable loading via pydantic-settings, command-line
argument parsing via argparse, validation, and produces a consolidated
immutable AppConfig instance for pipeline execution.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from pydantic import BaseModel, Field, field_validator, model_validator, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
err_console = Console(stderr=True)


class EnvSettings(BaseSettings):
    """Loads and validates configuration from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    obsidian_vault_path: Optional[Path] = Field(
        default=None,
        validation_alias=AliasChoices("OBSIDIAN_VAULT_PATH", "obsidian_vault_path"),
        description="Absolute path to the target Obsidian vault directory.",
    )
    api_base_url: str = Field(
        default="http://localhost:1234/v1",
        validation_alias=AliasChoices("API_BASE_URL", "api_base_url"),
        description="Base URL for the OpenAI-compatible local LLM server.",
    )
    api_key: str = Field(
        default="lm-studio",
        validation_alias=AliasChoices("API_KEY", "api_key"),
        description="API key for the LLM endpoint (placeholder for LM Studio).",
    )
    model_name: str = Field(
        default="local-model",
        validation_alias=AliasChoices("MODEL_NAME", "model_name"),
        description="Target model identifier in LM Studio (defaults to 'local-model').",
    )
    temperature: float = Field(
        default=0.2,
        validation_alias=AliasChoices("TEMPERATURE", "temperature"),
        description="Sampling temperature for the LLM generation.",
    )
    harness_path: Path = Field(
        default=Path("harness.md"),
        validation_alias=AliasChoices("HARNESS_PATH", "harness_path"),
        description="Path to the system prompt harness markdown file.",
    )


class AppConfig(BaseModel):
    """Consolidated, validated runtime configuration for the pipeline execution."""

    model_config = {"frozen": True}

    input_file: Path
    vault_path: Optional[Path] = None
    api_base_url: str
    api_key: str
    model_name: str
    temperature: float
    harness_path: Path
    output_filename: Optional[str] = None
    stream: bool = True
    dry_run: bool = False

    @field_validator("input_file")
    @classmethod
    def validate_input_file(cls, v: Path) -> Path:
        """Ensure input transcript file exists and is a readable file."""
        resolved = v.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Input transcript file not found: {v}")
        if not resolved.is_file():
            raise ValueError(f"Input transcript path is not a file: {v}")
        return resolved

    @field_validator("harness_path")
    @classmethod
    def validate_harness_path(cls, v: Path) -> Path:
        """Ensure prompt harness file exists and is a readable file."""
        resolved = v.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Prompt harness file not found: {v}")
        if not resolved.is_file():
            raise ValueError(f"Prompt harness path is not a file: {v}")
        return resolved

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Ensure temperature is within safe generation bounds."""
        if not (0.0 <= v <= 2.0):
            raise ValueError(f"Temperature must be between 0.0 and 2.0 (received: {v})")
        return v

    @field_validator("output_filename")
    @classmethod
    def normalize_output_filename(cls, v: Optional[str]) -> Optional[str]:
        """Ensure custom output filename ends with .md if provided."""
        if v is not None:
            v = v.strip()
            if not v:
                return None
            if not v.endswith(".md"):
                v = f"{v}.md"
        return v

    @model_validator(mode="after")
    def validate_vault_path(self) -> AppConfig:
        """Validate Obsidian vault directory if not in dry-run mode."""
        if not self.dry_run:
            if self.vault_path is None:
                raise ValueError(
                    "Obsidian vault path must be specified. Set OBSIDIAN_VAULT_PATH in .env "
                    "or provide it via the --vault / -v CLI argument."
                )
            resolved = self.vault_path.resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"Obsidian vault directory does not exist: {self.vault_path}")
            if not resolved.is_dir():
                raise ValueError(f"Obsidian vault path is not a directory: {self.vault_path}")
            # Overwrite with resolved path
            object.__setattr__(self, "vault_path", resolved)
        elif self.vault_path is not None:
            object.__setattr__(self, "vault_path", self.vault_path.resolve())

        return self


def create_cli_parser() -> argparse.ArgumentParser:
    """Build and configure the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="obsidian-notes",
        description="Transform raw transcripts into structured, Obsidian-ready notes using a local LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Path to the raw audio transcript file (.txt).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Custom output note filename (e.g., 'Calculus-Lecture-1.md'). "
        "If omitted, the title is extracted from the generated frontmatter.",
    )
    parser.add_argument(
        "-v",
        "--vault",
        type=Path,
        default=None,
        help="Path to target Obsidian vault folder (overrides OBSIDIAN_VAULT_PATH in .env).",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="Target model name in LM Studio (overrides MODEL_NAME in .env).",
    )
    parser.add_argument(
        "--harness",
        type=Path,
        default=None,
        help="Path to system prompt harness markdown file (overrides HARNESS_PATH in .env).",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (0.0 to 2.0, overrides TEMPERATURE in .env).",
    )
    parser.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        default=True,
        help="Disable real-time console streaming during note generation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run pipeline and stream/print generated note to console without writing to the Obsidian vault.",
    )

    return parser


def load_config(
    argv: Optional[Sequence[str]] = None,
    env_file: Optional[str] = ".env",
) -> AppConfig:
    """Load and merge configuration from CLI arguments, .env file, and defaults.

    Precedence order:
      1. Explicit CLI arguments
      2. Environment variables / .env file
      3. Built-in defaults

    Args:
        argv: Optional argument sequence for CLI parsing (defaults to sys.argv[1:]).
        env_file: Path to environment file (defaults to .env).

    Returns:
        A validated, immutable AppConfig instance.

    Raises:
        ValueError, FileNotFoundError: If validation fails.
    """
    parser = create_cli_parser()
    cli_args = parser.parse_args(argv)

    # Load environment settings
    env_settings = (
        EnvSettings(_env_file=env_file)
        if env_file
        else EnvSettings()
    )

    # CLI arguments take precedence over environment settings
    vault_path = cli_args.vault if cli_args.vault is not None else env_settings.obsidian_vault_path
    model_name = cli_args.model if cli_args.model is not None else env_settings.model_name
    temperature = cli_args.temperature if cli_args.temperature is not None else env_settings.temperature
    harness_path = cli_args.harness if cli_args.harness is not None else env_settings.harness_path

    return AppConfig(
        input_file=cli_args.input,
        vault_path=vault_path,
        api_base_url=env_settings.api_base_url,
        api_key=env_settings.api_key,
        model_name=model_name,
        temperature=temperature,
        harness_path=harness_path,
        output_filename=cli_args.output,
        stream=cli_args.stream,
        dry_run=cli_args.dry_run,
    )


def print_config_summary(config: AppConfig) -> None:
    """Render an aesthetic rich table displaying the active configuration."""
    table = Table(
        title="[bold cyan]Pipeline Configuration[/bold cyan]",
        title_justify="left",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Setting", style="dim", width=20)
    table.add_column("Value", style="bold white")

    table.add_row("Input Transcript", str(config.input_file))
    table.add_row(
        "Vault Path",
        str(config.vault_path) if config.vault_path else "[italic yellow]None (Dry Run)[/italic yellow]",
    )
    table.add_row("LM Studio Endpoint", config.api_base_url)
    table.add_row("Target Model", config.model_name)
    table.add_row("Temperature", str(config.temperature))
    table.add_row("Prompt Harness", str(config.harness_path))
    table.add_row(
        "Output Filename",
        config.output_filename if config.output_filename else "[italic dim]Auto (from frontmatter)[/italic dim]",
    )
    table.add_row("Streaming Enabled", "✓ Yes" if config.stream else "✗ No")
    table.add_row("Dry Run Mode", "✓ Active" if config.dry_run else "✗ Inactive")

    console.print(Panel(table, border_style="cyan", expand=False))


if __name__ == "__main__":
    try:
        app_config = load_config()
        print_config_summary(app_config)
    except Exception as exc:
        err_console.print(f"[bold red]Configuration Error:[/bold red] {exc}")
        sys.exit(1)
