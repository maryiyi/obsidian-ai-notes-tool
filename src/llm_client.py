"""LLM Client integration module for obsidian-ai-notes-tool.

Wraps the OpenAI-compatible local API exposed by LM Studio, providing
connection verification, robust error handling, and real-time token streaming.
"""

from __future__ import annotations

import sys
from typing import Callable, Iterator, Optional

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    BadRequestError,
    OpenAI,
)
from rich.console import Console

console = Console()
err_console = Console(stderr=True)


class LLMClientError(Exception):
    """Base exception for LLM client operations."""


class LLMConnectionError(LLMClientError):
    """Raised when the client cannot connect to the local LM Studio server."""


class LLMTimeoutError(LLMClientError):
    """Raised when the LLM request times out during generation."""


class LLMGenerationError(LLMClientError):
    """Raised when the LLM returns an error during inference."""


class LMStudioClient:
    """Client for interfacing with LM Studio's local OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "lm-studio",
        model_name: str = "local-model",
        timeout: float = 600.0,
    ) -> None:
        """Initialize the LM Studio client.

        Args:
            base_url: Base URL for the OpenAI-compatible server (e.g. http://localhost:1234/v1).
            api_key: API key string (placeholder for LM Studio).
            model_name: Target model identifier. Defaults to 'local-model'.
            timeout: Maximum timeout in seconds for API requests. Defaults to 600s.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout

        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

    def check_connection(self) -> bool:
        """Check whether the local LM Studio server is reachable.

        Returns:
            True if connection succeeded, False otherwise.
        """
        try:
            self._client.models.list()
            return True
        except Exception:
            return False

    def verify_connection_or_raise(self) -> None:
        """Verify server connectivity or raise a descriptive LLMConnectionError.

        Raises:
            LLMConnectionError: If the server cannot be reached with guidance on how to fix.
        """
        try:
            self._client.models.list()
        except APIConnectionError as exc:
            raise LLMConnectionError(
                f"Could not connect to LM Studio at '{self.base_url}'.\n"
                "Please ensure that:\n"
                "  1. LM Studio is running.\n"
                "  2. A model is loaded into memory.\n"
                "  3. The local server is started on the 'Local Server' tab (default: http://localhost:1234)."
            ) from exc
        except Exception as exc:
            raise LLMConnectionError(
                f"Failed to communicate with LM Studio endpoint '{self.base_url}': {exc}"
            ) from exc

    def stream_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
    ) -> Iterator[str]:
        """Send a prompt to LM Studio and yield text chunks in real-time as they arrive.

        Args:
            system_prompt: The system prompt harness instructions.
            user_prompt: The user prompt containing the transcript.
            temperature: Sampling temperature for generation.

        Yields:
            Token text strings as received from the model.

        Raises:
            LLMConnectionError: If connection to LM Studio drops.
            LLMTimeoutError: If generation times out.
            LLMGenerationError: If the server encounters an error during inference.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            stream = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content

        except APIConnectionError as exc:
            raise LLMConnectionError(
                f"Connection to LM Studio at '{self.base_url}' was lost during generation."
            ) from exc
        except APITimeoutError as exc:
            raise LLMTimeoutError(
                f"Generation timed out after {self.timeout}s waiting for LM Studio response."
            ) from exc
        except BadRequestError as exc:
            raise LLMGenerationError(
                f"LM Studio rejected the request (possible context limit exceeded): {exc}"
            ) from exc
        except APIError as exc:
            raise LLMGenerationError(f"LM Studio API returned an error: {exc}") from exc
        except Exception as exc:
            raise LLMClientError(f"Unexpected error during generation: {exc}") from exc

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        stream: bool = True,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Execute note generation, optionally streaming tokens to console or callback.

        Args:
            system_prompt: The system prompt harness instructions.
            user_prompt: The user prompt containing the transcript.
            temperature: Sampling temperature for generation.
            stream: Whether to stream tokens in real-time.
            on_token: Optional callback invoked for each streamed token chunk.
                      If stream=True and on_token is None, tokens are printed
                      directly to sys.stdout with immediate flushing.

        Returns:
            The complete generated note content as a string.

        Raises:
            LLMClientError: If generation fails.
        """
        if stream:
            collected_chunks: list[str] = []

            for token in self.stream_generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
            ):
                collected_chunks.append(token)
                if on_token is not None:
                    on_token(token)
                else:
                    sys.stdout.write(token)
                    sys.stdout.flush()

            # Ensure a trailing newline after streaming finishes
            if on_token is None and collected_chunks:
                sys.stdout.write("\n")
                sys.stdout.flush()

            return "".join(collected_chunks)

        # Non-streaming path
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                stream=False,
            )
            content = response.choices[0].message.content
            return content or ""
        except APIConnectionError as exc:
            raise LLMConnectionError(
                f"Could not connect to LM Studio at '{self.base_url}': {exc}"
            ) from exc
        except APITimeoutError as exc:
            raise LLMTimeoutError(f"Generation timed out after {self.timeout}s.") from exc
        except Exception as exc:
            raise LLMGenerationError(f"Generation failed: {exc}") from exc
