"""Ollama local LLM HTTP client.

Provides optional code polishing (comments, formatting, minor fixes)
via a locally running Ollama instance. The system is designed to work
fully without Ollama — the LLM is never in the critical path.
"""

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "codellama"
DEFAULT_TIMEOUT = 30.0


class OllamaClient:
    """HTTP client for the Ollama REST API."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    async def is_available(self) -> bool:
        """Check if the Ollama server is reachable.

        Returns:
            True if the server responds, False otherwise.
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, OSError):
            return False

    async def generate(self, prompt: str, system: str = "") -> str | None:
        """Send a prompt to Ollama and return the generated text.

        Args:
            prompt: The user prompt.
            system: Optional system prompt.

        Returns:
            The generated text, or None if Ollama is unavailable or errors.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                return data.get("response", "")
        except httpx.TimeoutException:
            logger.warning("Ollama request timed out after %.1fs", self.timeout)
            return None
        except (httpx.ConnectError, OSError):
            logger.warning("Ollama server not reachable at %s", self.base_url)
            return None
        except httpx.HTTPStatusError as exc:
            logger.warning("Ollama returned HTTP %d: %s", exc.response.status_code, exc)
            return None

    async def polish_code(self, code: str, framework: str = "pytorch") -> str:
        """Use the LLM to add comments and improve formatting.

        This is a best-effort operation. If Ollama is unavailable or returns
        garbage, the original code is returned unchanged.

        Args:
            code: The generated model source code.
            framework: 'pytorch' or 'keras'.

        Returns:
            Polished code, or the original code if LLM is unavailable.
        """
        system_prompt = (
            "You are a Python code formatter. You receive auto-generated "
            f"{framework} model code. Your job is to:\n"
            "1. Add clear docstrings and inline comments.\n"
            "2. Improve formatting if needed.\n"
            "3. Do NOT change the logic, layer order, or parameters.\n"
            "4. Return ONLY the Python code, no markdown fences or explanations."
        )
        prompt = f"Polish this {framework} model code:\n\n{code}"

        result = await self.generate(prompt, system=system_prompt)

        if result is None:
            logger.info("Ollama unavailable — returning original code")
            return code

        # Basic sanity: result should contain 'import' and 'class' or 'def'
        if "import" in result and ("class " in result or "def " in result):
            return result

        logger.warning("Ollama response did not look like valid code — using original")
        return code
