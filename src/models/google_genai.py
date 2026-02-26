"""
Google Generative AI (Gemini) Model Wrapper
Supports both Google AI Studio (API key) and Vertex AI (service account)
"""

import time
import os
from typing import Optional, List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core import exceptions as google_exceptions
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class GoogleGenAI:
    """Wrapper for Google's Generative AI (Gemini) models"""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        use_vertex_ai: bool = False,
        project_id: Optional[str] = None,
        location: str = "us-central1"  # ← changed from "global"
    ):
        self.model_name = model_name
        self.use_vertex_ai = use_vertex_ai or os.getenv("USE_VERTEX_AI", "").lower() == "true"

        if self.use_vertex_ai:
            self._init_vertex_ai(project_id, location)
        else:
            self._init_ai_studio(api_key)

    def _init_ai_studio(self, api_key: Optional[str]):
        """Initialize with Google AI Studio (API key)"""
        try:
            import google.generativeai as genai
            self.genai = genai
            self.backend = "AI Studio"

            self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not self.api_key:
                logger.warning("GEMINI_API_KEY or GOOGLE_API_KEY not found. Model calls will fail.")
            else:
                genai.configure(api_key=self.api_key)

            if not self.model_name.startswith("models/"):
                self.model_name = f"models/{self.model_name}"

            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"Initialized Google AI Studio with model: {self.model_name}")

        except ImportError:
            logger.error("google-generativeai not installed. Run: pip install google-generativeai")
            raise

    def _init_vertex_ai(self, project_id: Optional[str], location: str):
        """Initialize with Vertex AI (service account)"""
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
            self.backend = "Vertex AI"

            self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
            self.location = location

            if not self.project_id:
                logger.warning("GOOGLE_CLOUD_PROJECT not set. Model calls will fail.")

            # Vertex AI doesn't use 'models/' prefix
            if self.model_name.startswith("models/"):
                self.model_name = self.model_name.replace("models/", "")

            vertexai.init(project=self.project_id, location=self.location)
            self.model = GenerativeModel(self.model_name)
            logger.info(f"Initialized Vertex AI: model={self.model_name}, project={self.project_id}, location={self.location}")

        except ImportError:
            logger.error("google-cloud-aiplatform not installed. Run: pip install google-cloud-aiplatform")
            raise

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_output_tokens: int = 2048,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        if self.use_vertex_ai:
            return self._generate_vertex_ai(prompt, temperature, max_output_tokens, stop_sequences)
        else:
            return self._generate_ai_studio(prompt, temperature, max_output_tokens, stop_sequences)

    def _generate_ai_studio(self, prompt, temperature, max_output_tokens, stop_sequences):
        """Generate using Google AI Studio"""
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")

        generation_config = self.genai.types.GenerationConfig(
            candidate_count=1,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            stop_sequences=stop_sequences
        )
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            return response.text
        except Exception as e:
            logger.error(f"AI Studio generation error: {e}")
            if "429" in str(e):
                logger.warning("Rate limit hit, waiting 60s...")
                time.sleep(60)
                response = self.model.generate_content(prompt, generation_config=generation_config)
                return response.text
            raise

    @retry(
        retry=retry_if_exception_type(google_exceptions.ResourceExhausted),
        wait=wait_exponential(multiplier=2, min=30, max=180),  # 30s, 60s, 120s, 180s
        stop=stop_after_attempt(6)
    )
    def _generate_vertex_ai(self, prompt, temperature, max_output_tokens, stop_sequences):
        """Generate using Vertex AI with automatic retry on 429"""
        from vertexai.generative_models import GenerationConfig, SafetySetting, HarmCategory, HarmBlockThreshold

        if not self.project_id:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT is not set")

        generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            stop_sequences=stop_sequences or []
        )
        safety_settings = [
            SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT,        threshold=HarmBlockThreshold.BLOCK_NONE),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,       threshold=HarmBlockThreshold.BLOCK_NONE),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.BLOCK_NONE),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.BLOCK_NONE),
        ]
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            return response.text
        except google_exceptions.ResourceExhausted as e:
            logger.warning(f"Vertex AI 429 - will retry with backoff: {e}")
            raise  # tenacity handles the retry
        except Exception as e:
            logger.error(f"Vertex AI generation error: {e}")
            raise