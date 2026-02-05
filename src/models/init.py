"""Models module initialization"""

from src.models.prompt_manager import PromptManager, SpecializedPrompts
from src.models.prompt_templates import (
    PromptType,
    SQLPatterns,
    SQLValidationRules,
    PromptExamples
)

# Backward compatibility
TemplateManager = PromptManager

__all__ = [
    'PromptManager',
    'TemplateManager',  # Alias for backward compatibility
    'SpecializedPrompts',
    'PromptType',
    'SQLPatterns',
    'SQLValidationRules',
    'PromptExamples'
]