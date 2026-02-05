"""
Stub for template manager (if not available).
Provides basic functionality when template_manager is not installed.
"""

from enum import Enum
from typing import Dict


class PromptType(Enum):
    """Prompt types"""
    BASIC = "basic"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    RULE_BASED = "rule_based"
    ENHANCED = "enhanced"
    STEP_BY_STEP = "step_by_step"


class TemplateManager:
    """Basic template manager stub"""
    
    def __init__(self):
        self.common_patterns = {
            'count': 'SELECT COUNT(*) FROM table',
            'list all': 'SELECT * FROM table',
            'show': 'SELECT * FROM table',
        }
    
    def get_template(self, prompt_type: PromptType) -> str:
        """Get template for prompt type"""
        return """You are a SQL expert. Generate correct SQL queries.

Database Schema: {schema}

Rules:
1. Use simple, correct SQL
2. No aliases unless necessary
3. Return only SQL query

Question: {question}

SQL:"""


class SpecializedPrompts:
    """Specialized prompts stub"""
    
    def get_debugging_prompt(self) -> str:
        """Get debugging prompt"""
        return """Fix this SQL query.

Schema: {schema}
Question: {question}
Incorrect SQL: {sql}
Error: {error}

Corrected SQL:"""


__all__ = ['PromptType', 'TemplateManager', 'SpecializedPrompts']