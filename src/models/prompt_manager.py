"""
Prompt template manager for SQL generation.
Manages different prompt strategies and custom template creation.
"""

from typing import Dict, Optional
from src.models.prompt_templates import (
    PromptType,
    BasePromptTemplate,
    BasicPromptTemplate,
    FewShotPromptTemplate,
    ChainOfThoughtPromptTemplate,
    RuleBasedPromptTemplate,
    EnhancedPromptTemplate,
    StepByStepPromptTemplate,
    SQLPatterns,
    SQLValidationRules,
    PromptExamples
)
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class PromptManager:
    """Manages SQL generation prompt templates"""
    
    def __init__(self):
        self.templates: Dict[PromptType, BasePromptTemplate] = {
            PromptType.BASIC: BasicPromptTemplate(),
            PromptType.FEW_SHOT: FewShotPromptTemplate(),
            PromptType.CHAIN_OF_THOUGHT: ChainOfThoughtPromptTemplate(),
            PromptType.RULE_BASED: RuleBasedPromptTemplate(),
            PromptType.ENHANCED: EnhancedPromptTemplate(),
            PromptType.STEP_BY_STEP: StepByStepPromptTemplate()
        }
        
        self.patterns = SQLPatterns()
        self.rules = SQLValidationRules()
        self.examples = PromptExamples()
    
    def get_prompt(
        self,
        prompt_type: PromptType,
        schema: str,
        question: str,
        **kwargs
    ) -> str:
        """
        Get formatted prompt for given type
        
        Args:
            prompt_type: Type of prompt template
            schema: Database schema description
            question: Natural language question
            **kwargs: Additional template parameters
            
        Returns:
            Formatted prompt string
        """
        if prompt_type not in self.templates:
            logger.warning(f"Unknown prompt type: {prompt_type}, using BASIC")
            prompt_type = PromptType.BASIC
        
        template = self.templates[prompt_type]
        return template.format(schema=schema, question=question, **kwargs)
    
    def create_custom_prompt(
        self,
        schema: str,
        question: str,
        include_examples: bool = True,
        include_rules: bool = True,
        include_patterns: bool = True,
        include_reasoning: bool = False,
        max_examples: int = 5
    ) -> str:
        """
        Create a custom prompt with specified components
        
        Args:
            schema: Database schema description
            question: Natural language question
            include_examples: Include few-shot examples
            include_rules: Include validation rules
            include_patterns: Include common patterns
            include_reasoning: Include reasoning steps
            max_examples: Maximum number of examples to include
            
        Returns:
            Custom formatted prompt
        """
        template_parts = [
            "You are an expert SQL developer. Convert natural language questions to accurate SQL queries."
        ]
        
        if include_rules:
            template_parts.append(f"\n## VALIDATION RULES:\n{self.rules.get_rules_text()}")
        
        if include_patterns:
            template_parts.append(f"\n## COMMON PATTERNS:\n{self.patterns.get_patterns_text()}")
        
        if include_examples:
            template_parts.append(f"\n{self.examples.format_examples(max_examples)}")
        
        if include_reasoning:
            template_parts.append("""
## REASONING PROCESS:
1. Analyze the question to understand the request
2. Identify required tables and columns from the schema
3. Determine necessary JOIN conditions
4. Apply appropriate WHERE filters
5. Add GROUP BY for aggregations if needed
6. Include ORDER BY and LIMIT if specified
7. Validate the final query against schema and rules
""")
        
        # Always include critical rules
        template_parts.append(f"\n## CRITICAL RULES (MUST FOLLOW):\n{self.rules.get_critical_rules_text()}")
        
        template_parts.extend([
            "\nDatabase Schema:",
            schema,
            "\nQuestion:",
            question,
            "\nSQL Query:"
        ])
        
        return "\n".join(template_parts)
    
    def get_available_types(self) -> list:
        """Get list of available prompt types"""
        return list(self.templates.keys())


class SpecializedPrompts:
    """Specialized prompts for debugging and optimization"""
    
    @staticmethod
    def get_debugging_prompt(
        question: str,
        schema: str,
        sql: str,
        error: str
    ) -> str:
        """
        Get prompt for debugging SQL queries
        
        Args:
            question: Original question
            schema: Database schema
            sql: Problematic SQL query
            error: Error message
            
        Returns:
            Debugging prompt
        """
        return f"""You are an SQL debugging expert. Fix the provided SQL query to work correctly with the given schema.

Original Question: {question}
Database Schema: {schema}
Problematic SQL: {sql}
Error Message: {error}

Please provide:
1. **Error Analysis**: What's wrong with the query?
2. **Corrected SQL**: The fixed version
3. **Explanation**: Why the fix works

Corrected SQL Query:"""
    
    @staticmethod
    def get_optimization_prompt(
        question: str,
        schema: str,
        sql: str
    ) -> str:
        """
        Get prompt for SQL query optimization
        
        Args:
            question: Original question
            schema: Database schema
            sql: Current SQL query
            
        Returns:
            Optimization prompt
        """
        return f"""You are an SQL optimization expert. Improve the given query for better performance while maintaining correctness.

Original Question: {question}
Database Schema: {schema}
Current SQL: {sql}

Please provide:
1. **Performance Analysis**: Potential issues with current query
2. **Optimized SQL**: Improved version
3. **Optimization Explanation**: What improvements were made

Optimized SQL Query:"""
    
    @staticmethod
    def get_explanation_prompt(
        question: str,
        schema: str,
        sql: str
    ) -> str:
        """
        Get prompt for explaining SQL queries
        
        Args:
            question: Original question
            schema: Database schema
            sql: SQL query to explain
            
        Returns:
            Explanation prompt
        """
        return f"""You are an SQL education expert. Explain the given SQL query in clear, understandable terms.

Question: {question}
Database Schema: {schema}
SQL Query: {sql}

Please provide:
1. **Query Breakdown**: Explain each part of the query
2. **Logic Flow**: How the query works step by step
3. **Result Description**: What the query returns

Explanation:"""


# Convenience function for backward compatibility
def get_template_manager() -> PromptManager:
    """Get a PromptManager instance"""
    return PromptManager()