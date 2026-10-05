#!/usr/bin/env python3
"""Main orchestration script for obsidian-ai-notes-tool.

Transforms raw, unstructured audio transcripts into structured, Obsidian-ready
Markdown notes via local LLMs running in LM Studio.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from src.config import AppConfig, load_config, print_config_summary
from src.llm_client import (
    LLMClientError,
    LLMConnectionError,
    LLMTimeoutError,
    LMStudioClient,
)
from src.obsidian_io import (
    ObsidianIOError,
    extract_note_title,
    sanitize_filename,
    sanitize_llm_output,
    write_note_to_vault,
)
from src.prompt_harness import (
    PromptHarnessError,
    build_system_prompt,
    format_user_prompt,
    load_harness,
)

console = Console()
err_console = Console(stderr=True)


def print_banner() -> None:
    """Display application startup banner."""
    banner_text = Text()
    banner_text.append(" Obsidian AI Notes Tool ", style="bold white on blue")
    banner_text.append(" — Speech Transcript to Structured Notes", style="italic cyan")
    console.print(Panel(banner_text, border_style="blue", expand=False))


def generate_with_live_feedback(
    client: LMStudioClient,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    stream: bool = True,
) -> str:
    """Execute LLM generation with a rich status spinner transitioning to streaming.

    Displays a rich.status spinner while awaiting the model's first token.
    As soon as inference starts producing tokens, the spinner stops and tokens
    are streamed to stdout in real-time.

    Args:
        client: Initialized LMStudioClient instance.
        system_prompt: Prepared harness system prompt.
        user_prompt: Formatted user prompt containing the transcript.
        temperature: Sampling temperature.
        stream: Whether to stream tokens live to stdout.

    Returns:
        The complete, accumulated raw output from the LLM.

    Raises:
        LLMClientError: If generation fails or returns empty output.
    """
    collected_tokens: list[str] = []
    first_token_received = False

    with console.status(
        "[bold cyan]Contacting LM Studio and awaiting first token...[/bold cyan]",
        spinner="dots",
    ) as status:
        token_stream = client.stream_generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
        )

        for token in token_stream:
            if not first_token_received:
                # Stop spinner before writing the first token
                status.stop()
                first_token_received = True
                if stream:
                    console.print("[dim]First token received. Streaming note generation:[/dim]\n")

            collected_tokens.append(token)
            if stream:
                sys.stdout.write(token)
                sys.stdout.flush()

    if stream and collected_tokens:
        sys.stdout.write("\n\n")
        sys.stdout.flush()

    if not collected_tokens:
        raise LLMClientError("The model returned an empty response.")

    return "".join(collected_tokens)


def run_pipeline(config: Optional[AppConfig] = None) -> None:
    """Run the transcript-to-Obsidian note generation pipeline.

    Args:
        config: Optional pre-loaded AppConfig. If None, loaded from CLI and .env.
    """
    start_time = time.perf_counter()

    # 1. Load & validate configuration
    if config is None:
        config = load_config()

    print_banner()
    print_config_summary(config)

    # 2. Read input transcript
    console.print(f"[bold]Reading input transcript:[/bold] [cyan]{config.input_file}[/cyan]")
    try:
        transcript_text = config.input_file.read_text(encoding="utf-8").strip()
    except Exception as exc:
        raise FileNotFoundError(f"Failed to read transcript file '{config.input_file}': {exc}") from exc

    if not transcript_text:
        raise ValueError(f"Input transcript file '{config.input_file}' is empty.")

    word_count = len(transcript_text.split())
    console.print(f"[dim]Loaded transcript ({word_count:,} words, {len(transcript_text):,} characters).[/dim]\n")

    # 3. Load & format prompt harness
    console.print(f"[bold]Loading system prompt harness:[/bold] [cyan]{config.harness_path}[/cyan]")
    raw_harness = load_harness(config.harness_path)
    system_prompt = build_system_prompt(raw_harness)
    user_prompt = format_user_prompt(transcript_text, filename=config.input_file.name)

    # 4. Initialize LLM client and verify server connectivity
    client = LMStudioClient(
        base_url=config.api_base_url,
        api_key=config.api_key,
        model_name=config.model_name,
    )

    console.print(f"[bold]Verifying connection to LM Studio[/bold] ([dim]{config.api_base_url}[/dim])...")
    client.verify_connection_or_raise()
    console.print("[bold green]✓ Connected to LM Studio successfully![/bold green]\n")

    # 5. Execute generation
    raw_output = generate_with_live_feedback(
        client=client,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=config.temperature,
        stream=config.stream,
    )

    # 6. Post-processing: clean hallucinated fences
    cleaned_note = sanitize_llm_output(raw_output)

    # 7. Note naming and extraction
    extracted_title = extract_note_title(cleaned_note, fallback_title=config.input_file.stem)
    target_filename = config.output_filename or sanitize_filename(extracted_title)

    elapsed_seconds = time.perf_counter() - start_time

    # 8. Vault Output or Dry-Run
    if config.dry_run:
        summary_panel = Panel(
            f"[bold yellow]DRY RUN COMPLETE (No files written)[/bold yellow]\n\n"
            f"[bold]Detected Title:[/bold] {extracted_title}\n"
            f"[bold]Intended Filename:[/bold] {target_filename}\n"
            f"[bold]Generation Time:[/bold] {elapsed_seconds:.1f}s\n"
            f"[bold]Output Length:[/bold] {len(cleaned_note):,} characters",
            title="[bold yellow]Dry Run Summary[/bold yellow]",
            border_style="yellow",
        )
        console.print(summary_panel)
        return

    # Write note safely to Obsidian vault
    assert config.vault_path is not None, "Vault path must be set for non-dry-run mode."
    created_path = write_note_to_vault(
        vault_path=config.vault_path,
        content=cleaned_note,
        desired_filename=config.output_filename,
        fallback_title=config.input_file.stem,
        overwrite=False,
    )

    # Success presentation
    success_panel = Panel(
        f"[bold green]✓ Note successfully created in Obsidian Vault![/bold green]\n\n"
        f"[bold]File:[/bold] [link=file://{created_path}]{created_path}[/link]\n"
        f"[bold]Title:[/bold] {extracted_title}\n"
        f"[bold]File Size:[/bold] {created_path.stat().st_size:,} bytes\n"
        f"[bold]Total Time:[/bold] {elapsed_seconds:.1f}s",
        title="[bold green]Pipeline Success[/bold green]",
        border_style="green",
    )
    console.print(success_panel)


def main() -> None:
    """CLI entrypoint with clean exception handling and graceful interruption."""
    try:
        run_pipeline()
    except KeyboardInterrupt:
        err_console.print(
            "\n[bold yellow]⚠ Process interrupted by user (Ctrl+C). "
            "No files were written to the Obsidian vault.[/bold yellow]"
        )
        sys.exit(130)
    except LLMConnectionError as exc:
        err_console.print(f"\n[bold red]LM Studio Connection Error:[/bold red]\n{exc}")
        sys.exit(1)
    except (LLMClientError, PromptHarnessError, ObsidianIOError, ValueError, FileNotFoundError) as exc:
        err_console.print(f"\n[bold red]Error:[/bold red] {exc}")
        sys.exit(1)
    except Exception as exc:
        err_console.print(f"\n[bold red]Unexpected Failure:[/bold red] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
