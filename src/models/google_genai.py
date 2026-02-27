"""
Google Generative AI Model Wrapper
Supports:
  - Google AI Studio (API key)
  - Vertex AI Gemini (ADC)
  - Vertex AI MaaS / DeepSeek (requests-based, no openai pkg needed)
"""

import re
import time
import os
import subprocess
from typing import Optional, List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core import exceptions as google_exceptions
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class GoogleGenAI:
    """Wrapper for Google's Generative AI models (Gemini + DeepSeek MaaS)"""

    def __init__(
        self,
        model_name: str = "deepseek-ai/deepseek-r1-0528-maas",
        api_key: Optional[str] = None,
        use_vertex_ai: bool = False,
        project_id: Optional[str] = None,
        location: str = "us-central1"
    ):
        self.model_name = model_name
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location
        self.use_vertex_ai = use_vertex_ai or os.getenv("USE_VERTEX_AI", "").lower() == "true"

        if self._is_maas_model(model_name):
            self._init_maas()
        elif self.use_vertex_ai:
            self._init_vertex_ai(project_id, location)
        else:
            self._init_ai_studio(api_key)

    def _is_maas_model(self, model_name: str) -> bool:
        maas_prefixes = ("deepseek-ai/", "meta/", "mistral-ai/", "anthropic/")
        return any(model_name.startswith(p) for p in maas_prefixes)

    # ------------------------------------------------------------------ #
    #  MaaS backend — pure requests, no openai package needed             #
    # ------------------------------------------------------------------ #
    def _init_maas(self):
        import requests  # stdlib-adjacent, always available
        self.backend = "Vertex AI MaaS"
        self._requests = requests
        if not self.project_id:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT is not set")
        self._maas_url = (
            f"https://{self.location}-aiplatform.googleapis.com/v1/"
            f"projects/{self.project_id}/locations/{self.location}"
            f"/endpoints/openapi/chat/completions"
        )
        self._token = self._get_access_token()
        self._token_expiry = time.time() + 3500
        logger.info(f"Initialized Vertex AI MaaS: model={self.model_name}, project={self.project_id}")

    def _get_access_token(self) -> str:
        """Get ADC access token — tries gcloud first, falls back to google-auth"""
        try:
            result = subprocess.run(
                ["gcloud", "auth", "application-default", "print-access-token"],
                capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except Exception:
            pass
        try:
            import google.auth
            import google.auth.transport.requests
            creds, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            creds.refresh(google.auth.transport.requests.Request())
            return creds.token
        except Exception as e:
            raise RuntimeError(f"Could not obtain GCP access token: {e}")

    def _refresh_token_if_needed(self):
        if time.time() > self._token_expiry:
            self._token = self._get_access_token()
            self._token_expiry = time.time() + 3500
            logger.info("Refreshed MaaS access token")

    def _wrap_prompt_for_maas(self, prompt: str) -> list:
        """Wrap prompt as chat messages with strict SQL-only instruction"""
        system = (
            "You are a SQL query generator. You output ONLY SQL. "
            "ABSOLUTE RULES - no exceptions:\n"
            "1. Output a single SQL query and nothing else\n"
            "2. No explanations, no reasoning, no comments\n"
            "3. No markdown, no code fences, no backticks\n"
            "4. No 'But', 'However', 'Note', or any English text\n"
            "5. If unsure, output your best guess SQL - never output text\n"
            "6. Stop immediately after the semicolon or last SQL token\n"
            "Your entire response must be valid SQL starting with SELECT/WITH/INSERT/UPDATE/DELETE."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

    def _generate_maas(self, prompt: str, temperature: float, max_output_tokens: int) -> str:
        self._refresh_token_if_needed()
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }
        
        messages = self._wrap_prompt_for_maas(prompt)
        
        # Check if we used assistant pre-fill (last message is assistant role)
        has_prefill = messages and messages[-1].get("role") == "assistant"
        prefill_content = messages[-1].get("content", "") if has_prefill else ""
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }
        resp = self._requests.post(self._maas_url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"] or ""

        # Strip <think>...</think> reasoning blocks (DeepSeek R1)
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        # If we pre-filled the assistant turn with "SELECT", the model's response
        # is the continuation — prepend the pre-fill so extraction works normally.
        if has_prefill and prefill_content and not content.upper().startswith("SELECT"):
            content = prefill_content + " " + content

        return content

    # ------------------------------------------------------------------ #
    #  Google AI Studio backend                                           #
    # ------------------------------------------------------------------ #
    def _init_ai_studio(self, api_key: Optional[str]):
        try:
            import google.generativeai as genai
            self.genai = genai
            self.backend = "AI Studio"
            self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not self.api_key:
                logger.warning("GEMINI_API_KEY not found.")
            else:
                genai.configure(api_key=self.api_key)
            if not self.model_name.startswith("models/"):
                self.model_name = f"models/{self.model_name}"
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"Initialized Google AI Studio: {self.model_name}")
        except ImportError:
            raise RuntimeError("google-generativeai not installed.")

    def _generate_ai_studio(self, prompt, temperature, max_output_tokens, stop_sequences):
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        gen_cfg = self.genai.types.GenerationConfig(
            candidate_count=1, max_output_tokens=max_output_tokens,
            temperature=temperature, stop_sequences=stop_sequences
        )
        safety = [
            {"category": "HARM_CATEGORY_HARASSMENT",       "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH",      "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT","threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT","threshold": "BLOCK_NONE"},
        ]
        try:
            r = self.model.generate_content(prompt, generation_config=gen_cfg, safety_settings=safety)
            return r.text
        except Exception as e:
            if "429" in str(e):
                logger.warning("Rate limit hit, waiting 60s...")
                time.sleep(60)
                return self.model.generate_content(prompt, generation_config=gen_cfg).text
            raise

    # ------------------------------------------------------------------ #
    #  Vertex AI Gemini backend                                           #
    # ------------------------------------------------------------------ #
    def _init_vertex_ai(self, project_id: Optional[str], location: str):
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
            self.backend = "Vertex AI Gemini"
            self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
            self.location = location
            if self.model_name.startswith("models/"):
                self.model_name = self.model_name.replace("models/", "")
            vertexai.init(project=self.project_id, location=self.location)
            self.model = GenerativeModel(self.model_name)
            logger.info(f"Initialized Vertex AI Gemini: {self.model_name}")
        except ImportError:
            raise RuntimeError("google-cloud-aiplatform not installed.")

    @retry(
        retry=retry_if_exception_type(google_exceptions.ResourceExhausted),
        wait=wait_exponential(multiplier=2, min=30, max=180),
        stop=stop_after_attempt(6)
    )
    def _generate_vertex_ai(self, prompt, temperature, max_output_tokens, stop_sequences):
        from vertexai.generative_models import GenerationConfig, SafetySetting, HarmCategory, HarmBlockThreshold
        gen_cfg = GenerationConfig(
            temperature=temperature, max_output_tokens=max_output_tokens,
            stop_sequences=stop_sequences or []
        )
        safety = [
            SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT,        threshold=HarmBlockThreshold.BLOCK_NONE),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,       threshold=HarmBlockThreshold.BLOCK_NONE),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.BLOCK_NONE),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.BLOCK_NONE),
        ]
        try:
            r = self.model.generate_content(prompt, generation_config=gen_cfg, safety_settings=safety)
            return r.text
        except google_exceptions.ResourceExhausted as e:
            logger.warning(f"Vertex AI 429 - will retry: {e}")
            raise
        except Exception as e:
            logger.error(f"Vertex AI generation error: {e}")
            raise

    # ------------------------------------------------------------------ #
    #  Unified entry point                                                #
    # ------------------------------------------------------------------ #
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_output_tokens: int = 2048,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        if self._is_maas_model(self.model_name):
            return self._generate_maas(prompt, temperature, max_output_tokens)
        elif self.use_vertex_ai:
            return self._generate_vertex_ai(prompt, temperature, max_output_tokens, stop_sequences)
        else:
            return self._generate_ai_studio(prompt, temperature, max_output_tokens, stop_sequences)