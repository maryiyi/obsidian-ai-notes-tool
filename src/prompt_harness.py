"""System prompt harness manager for obsidian-ai-notes-tool.

Handles loading, validating, and parameterizing the harness.md prompt template
to instruct the local LLM on strict Obsidian note generation.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional


class PromptHarnessError(Exception):
    """Raised when the prompt harness cannot be loaded or processed."""


def load_harness(harness_path: Path) -> str:
    """Read the prompt harness template from the specified path.

    Args:
        harness_path: Absolute or relative path to the harness markdown file.

    Returns:
        The raw harness template text.

    Raises:
        PromptHarnessError: If the file is missing, cannot be read, or is empty.
    """
    path = harness_path.resolve()
    if not path.exists():
        raise PromptHarnessError(f"Prompt harness template not found: {path}")
    if not path.is_file():
        raise PromptHarnessError(f"Prompt harness path is not a file: {path}")

    try:
        content = path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        raise PromptHarnessError(f"Failed to read prompt harness from {path}: {exc}") from exc

    if not content:
        raise PromptHarnessError(f"Prompt harness file is empty: {path}")

    return content


def build_system_prompt(
    template: str,
    context: Optional[Mapping[str, Any]] = None,
) -> str:
    """Format the system prompt template with runtime variables.

    Replaces '{current_date}' and any custom key-value pairs provided in the context.

    Args:
        template: Raw harness markdown template string.
        context: Optional dictionary of variables to interpolate into the prompt.

    Returns:
        The prepared system prompt ready to send to the LLM.
    """
    variables: dict[str, str] = {
        "current_date": datetime.now().strftime("%Y-%m-%d"),
    }
    if context:
        for k, v in context.items():
            variables[k] = str(v)

    # Replace known placeholders safely without failing on unmatched curly braces (like in LaTeX)
    system_prompt = template
    for key, val in variables.items():
        placeholder = f"{{{key}}}"
        if placeholder in system_prompt:
            system_prompt = system_prompt.replace(placeholder, val)

    return system_prompt


def format_user_prompt(transcript_text: str, filename: Optional[str] = None) -> str:
    """Wrap raw transcript text into a structured user message.

    Args:
        transcript_text: Unstructured audio transcript content.
        filename: Optional source transcript filename for reference.

    Returns:
        Formatted user prompt to send to the model.
    """
    cleaned_transcript = transcript_text.strip()
    source_tag = f' source="{filename}"' if filename else ""

    return (
        "Generate a structured, comprehensive Obsidian Markdown note for the following "
        "raw lecture transcript according to your system instructions. "
        "Remember: output ONLY the markdown note starting with YAML frontmatter on line 1, "
        "with zero conversational filler or outer code block fences.\n\n"
        f"<TRANSCRIPT{source_tag}>\n"
        f"{cleaned_transcript}\n"
        "</TRANSCRIPT>"
    )
