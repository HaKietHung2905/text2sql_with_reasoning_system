"""
Google Generative AI (Gemini) Model Wrapper
Supports both Google AI Studio (API key) and Vertex AI (service account)
"""

import os
import time
from typing import Optional, List
from utils.logging_utils import get_logger

logger = get_logger(__name__)

class GoogleGenAI:
    """Wrapper for Google's Generative AI (Gemini) models
    
    Supports two authentication modes:
    1. Google AI Studio: Use API key (simpler, for development)
    2. Vertex AI: Use service account (production-grade)
    """
    
    def __init__(
        self, 
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        use_vertex_ai: bool = False,
        project_id: Optional[str] = None,
        location: str = "us-central1"
    ):
        """
        Initialize the Gemini model
        
        Args:
            model_name: Name of the Gemini model (e.g., 'gemini-1.5-flash', 'gemini-1.5-pro')
            api_key: Google AI Studio API key (for AI Studio mode)
            use_vertex_ai: If True, use Vertex AI instead of AI Studio
            project_id: Google Cloud project ID (for Vertex AI mode)
            location: GCP region for Vertex AI (default: us-central1)
        """
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
            
            # AI Studio uses 'models/' prefix
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
            self.vertexai = vertexai
            self.GenerativeModel = GenerativeModel
            self.backend = "Vertex AI"
            
            self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
            self.location = location
            
            if not self.project_id:
                logger.warning("GOOGLE_CLOUD_PROJECT not set. Model calls will fail.")
                logger.warning("Set GOOGLE_CLOUD_PROJECT env var or pass project_id parameter")
            
            # Vertex AI doesn't use 'models/' prefix
            if self.model_name.startswith("models/"):
                self.model_name = self.model_name.replace("models/", "")
            
            vertexai.init(project=self.project_id, location=self.location)
            self.model = GenerativeModel(self.model_name)
            # logger.info(f"Initialized Vertex AI with model: {self.model_name}, project: {self.project_id}")
            
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
        """
        Generate content from a prompt
        
        Args:
            prompt: Input text prompt
            temperature: Generation temperature (0.0 = deterministic)
            max_output_tokens: Max tokens to generate
            stop_sequences: List of strings to stop generation at
            
        Returns:
            Generated text content
        """
        if self.use_vertex_ai:
            return self._generate_vertex_ai(prompt, temperature, max_output_tokens, stop_sequences)
        else:
            return self._generate_ai_studio(prompt, temperature, max_output_tokens, stop_sequences)
    
    def _generate_ai_studio(
        self, 
        prompt: str, 
        temperature: float, 
        max_output_tokens: int, 
        stop_sequences: Optional[List[str]]
    ) -> str:
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
                logger.warning("Rate limit hit, waiting 2s...")
                time.sleep(2)
                try:
                    response = self.model.generate_content(prompt, generation_config=generation_config)
                    return response.text
                except Exception as retry_e:
                    logger.error(f"Retry failed: {retry_e}")
                    raise retry_e
            raise e
    
    def _generate_vertex_ai(
        self, 
        prompt: str, 
        temperature: float, 
        max_output_tokens: int, 
        stop_sequences: Optional[List[str]]
    ) -> str:
        """Generate using Vertex AI"""
        from vertexai.generative_models import GenerationConfig, SafetySetting, HarmCategory, HarmBlockThreshold
        
        if not self.project_id:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT is not set")

        generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            stop_sequences=stop_sequences or []
        )
        
        safety_settings = [
            SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=HarmBlockThreshold.BLOCK_NONE),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=HarmBlockThreshold.BLOCK_NONE),
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
            
        except Exception as e:
            logger.error(f"Vertex AI generation error: {e}")
            raise e
