"""
Tests for prompt manager and templates.
Comprehensive test suite for SQL prompt generation system.
"""

import pytest
from src.models.prompt_manager import PromptManager, SpecializedPrompts
from src.models.prompt_templates import (
    PromptType,
    SQLPatterns,
    SQLValidationRules,
    PromptExamples,
    BasePromptTemplate,
    BasicPromptTemplate,
    FewShotPromptTemplate,
    ChainOfThoughtPromptTemplate,
    RuleBasedPromptTemplate,
    EnhancedPromptTemplate,
    StepByStepPromptTemplate
)


class TestPromptManager:
    """Test suite for PromptManager"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.manager = PromptManager()
        self.test_schema = """Table: employees [id, name, department_id, salary]
Table: departments [id, name]"""
        self.test_question = "How many employees are in each department?"
    
    def test_initialization(self):
        """Test PromptManager initializes correctly"""
        assert self.manager is not None
        assert len(self.manager.templates) == 6
        assert hasattr(self.manager, 'patterns')
        assert hasattr(self.manager, 'rules')
        assert hasattr(self.manager, 'examples')
    
    def test_get_available_types(self):
        """Test getting available prompt types"""
        types = self.manager.get_available_types()
        
        assert len(types) == 6
        assert PromptType.BASIC in types
        assert PromptType.FEW_SHOT in types
        assert PromptType.CHAIN_OF_THOUGHT in types
        assert PromptType.RULE_BASED in types
        assert PromptType.ENHANCED in types
        assert PromptType.STEP_BY_STEP in types
    
    def test_basic_prompt_generation(self):
        """Test basic prompt generation"""
        prompt = self.manager.get_prompt(
            prompt_type=PromptType.BASIC,
            schema=self.test_schema,
            question=self.test_question
        )
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert self.test_schema in prompt
        assert self.test_question in prompt
        assert "CRITICAL RULES" in prompt
        assert "SQL Query:" in prompt
    
    def test_few_shot_prompt_generation(self):
        """Test few-shot prompt with examples"""
        prompt = self.manager.get_prompt(
            prompt_type=PromptType.FEW_SHOT,
            schema=self.test_schema,
            question=self.test_question,
            max_examples=3
        )
        
        assert "Example 1:" in prompt
        assert "Example 2:" in prompt
        assert "Example 3:" in prompt
        assert "Example 4:" not in prompt  # Should only have 3
        assert self.test_schema in prompt
        assert self.test_question in prompt
    
    def test_few_shot_prompt_with_max_examples(self):
        """Test few-shot prompt respects max_examples parameter"""
        prompt_3 = self.manager.get_prompt(
            prompt_type=PromptType.FEW_SHOT,
            schema=self.test_schema,
            question=self.test_question,
            max_examples=3
        )
        
        prompt_5 = self.manager.get_prompt(
            prompt_type=PromptType.FEW_SHOT,
            schema=self.test_schema,
            question=self.test_question,
            max_examples=5
        )
        
        assert len(prompt_5) > len(prompt_3)
        assert "Example 5:" in prompt_5
        assert "Example 5:" not in prompt_3
    
    def test_chain_of_thought_prompt(self):
        """Test chain-of-thought prompt generation"""
        prompt = self.manager.get_prompt(
            prompt_type=PromptType.CHAIN_OF_THOUGHT,
            schema=self.test_schema,
            question=self.test_question
        )
        
        assert "step-by-step reasoning" in prompt.lower()
        assert "Analyze the question" in prompt
        assert "Identify required tables" in prompt
        assert "Construct SQL" in prompt
        assert self.test_schema in prompt
        assert self.test_question in prompt
    
    def test_rule_based_prompt(self):
        """Test rule-based prompt generation"""
        prompt = self.manager.get_prompt(
            prompt_type=PromptType.RULE_BASED,
            schema=self.test_schema,
            question=self.test_question
        )
        
        assert "VALIDATION RULES" in prompt
        assert "COMMON QUESTION PATTERNS" in prompt
        assert "SCHEMA VALIDATION CHECKLIST" in prompt
        assert "ERROR HANDLING" in prompt
        assert self.test_schema in prompt
        assert self.test_question in prompt
    
    def test_enhanced_prompt(self):
        """Test enhanced prompt generation"""
        prompt = self.manager.get_prompt(
            prompt_type=PromptType.ENHANCED,
            schema=self.test_schema,
            question=self.test_question,
            max_examples=3
        )
        
        assert "Expert SQL Query Generator" in prompt
        assert "VALIDATION RULES" in prompt
        assert "COMMON QUESTION PATTERNS" in prompt
        assert "QUERY CONSTRUCTION PROCESS" in prompt
        assert "Example 1:" in prompt
        assert self.test_schema in prompt
        assert self.test_question in prompt
    
    def test_step_by_step_prompt(self):
        """Test step-by-step prompt generation"""
        prompt = self.manager.get_prompt(
            prompt_type=PromptType.STEP_BY_STEP,
            schema=self.test_schema,
            question=self.test_question
        )
        
        assert "Step 1: Question Decomposition" in prompt
        assert "Step 2: Schema Mapping" in prompt
        assert "Step 3: Query Structure Planning" in prompt
        assert "Step 4: SQL Construction" in prompt
        assert "Step 5: Validation Check" in prompt
        assert self.test_schema in prompt
        assert self.test_question in prompt
    
    def test_custom_prompt_creation_all_components(self):
        """Test custom prompt with all components"""
        prompt = self.manager.create_custom_prompt(
            schema=self.test_schema,
            question=self.test_question,
            include_examples=True,
            include_rules=True,
            include_patterns=True,
            include_reasoning=True,
            max_examples=3
        )
        
        assert "VALIDATION RULES" in prompt
        assert "COMMON PATTERNS" in prompt
        assert "Example 1:" in prompt
        assert "REASONING PROCESS" in prompt
        assert self.test_schema in prompt
        assert self.test_question in prompt
    
    def test_custom_prompt_creation_minimal(self):
        """Test custom prompt with minimal components"""
        prompt = self.manager.create_custom_prompt(
            schema=self.test_schema,
            question=self.test_question,
            include_examples=False,
            include_rules=False,
            include_patterns=False,
            include_reasoning=False
        )
        
        assert "VALIDATION RULES" not in prompt
        assert "COMMON PATTERNS" not in prompt
        assert "Example 1:" not in prompt
        assert "REASONING PROCESS" not in prompt
        # Critical rules should always be present
        assert "CRITICAL RULES" in prompt
        assert self.test_schema in prompt
        assert self.test_question in prompt
    
    def test_custom_prompt_partial_components(self):
        """Test custom prompt with selective components"""
        prompt = self.manager.create_custom_prompt(
            schema=self.test_schema,
            question=self.test_question,
            include_examples=True,
            include_rules=False,
            include_patterns=True,
            include_reasoning=False,
            max_examples=2
        )
        
        assert "VALIDATION RULES" not in prompt
        assert "COMMON PATTERNS" in prompt
        assert "Example 1:" in prompt
        assert "Example 2:" in prompt
        assert "Example 3:" not in prompt
    
    def test_invalid_prompt_type_fallback(self):
        """Test that invalid prompt type falls back to basic"""
        # This should not raise an error, should use BASIC
        prompt = self.manager.get_prompt(
            prompt_type="INVALID_TYPE",
            schema=self.test_schema,
            question=self.test_question
        )
        
        # Should still generate a valid prompt
        assert isinstance(prompt, str)
        assert len(prompt) > 0


class TestPromptTemplates:
    """Test suite for individual prompt templates"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.test_schema = "Table: test [id, name]"
        self.test_question = "Show all records"
    
    def test_basic_template(self):
        """Test BasicPromptTemplate"""
        template = BasicPromptTemplate()
        prompt = template.format(
            schema=self.test_schema,
            question=self.test_question
        )
        
        assert self.test_schema in prompt
        assert self.test_question in prompt
        # Fix: Check for exact case or use case-insensitive check properly
        assert "expert sql developer" in prompt.lower()  # This should work
        # OR be more specific:
        assert "You are an expert SQL developer" in prompt
    
    def test_few_shot_template(self):
        """Test FewShotPromptTemplate"""
        template = FewShotPromptTemplate()
        prompt = template.format(
            schema=self.test_schema,
            question=self.test_question,
            max_examples=2
        )
        
        assert "Example 1:" in prompt
        assert "Example 2:" in prompt
    
    def test_chain_of_thought_template(self):
        """Test ChainOfThoughtPromptTemplate"""
        template = ChainOfThoughtPromptTemplate()
        prompt = template.format(
            schema=self.test_schema,
            question=self.test_question
        )
        
        assert "reasoning process" in prompt.lower()
        assert "Analyze the question" in prompt
    
    def test_rule_based_template(self):
        """Test RuleBasedPromptTemplate"""
        template = RuleBasedPromptTemplate()
        prompt = template.format(
            schema=self.test_schema,
            question=self.test_question
        )
        
        assert "VALIDATION RULES" in prompt
        assert "COMMON QUESTION PATTERNS" in prompt
    
    def test_enhanced_template(self):
        """Test EnhancedPromptTemplate"""
        template = EnhancedPromptTemplate()
        prompt = template.format(
            schema=self.test_schema,
            question=self.test_question,
            max_examples=2
        )
        
        assert "Expert SQL Query Generator" in prompt
        assert "Example 1:" in prompt
    
    def test_step_by_step_template(self):
        """Test StepByStepPromptTemplate"""
        template = StepByStepPromptTemplate()
        prompt = template.format(
            schema=self.test_schema,
            question=self.test_question
        )
        
        assert "Step 1:" in prompt
        assert "Step 2:" in prompt


class TestSQLPatterns:
    """Test suite for SQLPatterns"""
    
    def test_patterns_exist(self):
        """Test that common patterns are defined"""
        patterns = SQLPatterns.PATTERNS
        
        assert len(patterns) > 0
        assert "How many" in patterns
        assert "List all" in patterns
        assert "Top N/Highest N" in patterns
    
    def test_get_patterns_text(self):
        """Test formatting patterns as text"""
        text = SQLPatterns.get_patterns_text()
        
        assert isinstance(text, str)
        assert "How many" in text
        assert "SELECT COUNT(*)" in text
        assert "→" in text


class TestSQLValidationRules:
    """Test suite for SQLValidationRules"""
    
    def test_rules_exist(self):
        """Test that validation rules are defined"""
        rules = SQLValidationRules.RULES
        
        assert len(rules) > 0
        assert any("Schema Compliance" in rule for rule in rules)
        assert any("Syntax Correctness" in rule for rule in rules)
    
    def test_critical_rules_exist(self):
        """Test that critical rules are defined"""
        critical = SQLValidationRules.CRITICAL_RULES
        
        assert len(critical) > 0
        assert any("single table" in rule.lower() for rule in critical)
    
    def test_get_rules_text(self):
        """Test formatting rules as text"""
        text = SQLValidationRules.get_rules_text()
        
        assert isinstance(text, str)
        assert "1." in text
        assert "Schema Compliance" in text
    
    def test_get_critical_rules_text(self):
        """Test formatting critical rules"""
        text = SQLValidationRules.get_critical_rules_text()
        
        assert isinstance(text, str)
        assert "single table" in text.lower()


class TestPromptExamples:
    """Test suite for PromptExamples"""
    
    def test_examples_exist(self):
        """Test that examples are defined"""
        examples = PromptExamples.EXAMPLES
        
        assert len(examples) > 0
        assert all('question' in ex for ex in examples)
        assert all('sql' in ex for ex in examples)
        assert all('schema' in ex for ex in examples)
    
    def test_format_examples_default(self):
        """Test formatting examples with defaults"""
        text = PromptExamples.format_examples()
        
        assert isinstance(text, str)
        assert "Example 1:" in text
        assert "Question:" in text
        assert "SQL:" in text
        assert "Explanation:" in text
    
    def test_format_examples_limit(self):
        """Test formatting limited number of examples"""
        text_3 = PromptExamples.format_examples(max_examples=3)
        text_5 = PromptExamples.format_examples(max_examples=5)
        
        assert "Example 3:" in text_3
        assert "Example 4:" not in text_3
        assert "Example 5:" in text_5
    
    def test_format_examples_no_explanation(self):
        """Test formatting without explanations"""
        text = PromptExamples.format_examples(
            max_examples=2,
            include_explanation=False
        )
        
        assert "Example 1:" in text
        assert "Question:" in text
        assert "SQL:" in text
        assert "Explanation:" not in text


class TestSpecializedPrompts:
    """Test suite for SpecializedPrompts"""
    
    def test_debugging_prompt(self):
        """Test debugging prompt generation"""
        prompt = SpecializedPrompts.get_debugging_prompt(
            question="Show all users",
            schema="Table: users [id, name]",
            sql="SELECT * FORM users",
            error="SQL syntax error near 'FORM'"
        )
        
        assert "debugging expert" in prompt.lower()
        assert "Error Analysis" in prompt
        assert "Corrected SQL" in prompt
        assert "Explanation" in prompt
        assert "FORM" in prompt
        assert "syntax error" in prompt
    
    def test_optimization_prompt(self):
        """Test optimization prompt generation"""
        prompt = SpecializedPrompts.get_optimization_prompt(
            question="Find highest paid employee",
            schema="Table: employees [id, name, salary]",
            sql="SELECT * FROM employees ORDER BY salary DESC"
        )
        
        assert "optimization expert" in prompt.lower()
        assert "Performance Analysis" in prompt
        assert "Optimized SQL" in prompt
        assert "Optimization Explanation" in prompt
        assert "ORDER BY salary DESC" in prompt
    
    def test_explanation_prompt(self):
        """Test explanation prompt generation"""
        prompt = SpecializedPrompts.get_explanation_prompt(
            question="Count employees by department",
            schema="Table: employees [id, name, dept_id]",
            sql="SELECT dept_id, COUNT(*) FROM employees GROUP BY dept_id"
        )
        
        assert "education expert" in prompt.lower()
        assert "Query Breakdown" in prompt
        assert "Logic Flow" in prompt
        assert "Result Description" in prompt
        assert "GROUP BY dept_id" in prompt


class TestIntegration:
    """Integration tests for prompt system"""
    
    def test_full_workflow(self):
        """Test complete workflow from manager to formatted prompt"""
        manager = PromptManager()
        
        # Test each prompt type
        for prompt_type in [PromptType.BASIC, PromptType.FEW_SHOT, 
                           PromptType.ENHANCED]:
            prompt = manager.get_prompt(
                prompt_type=prompt_type,
                schema="Table: test [id]",
                question="Test question"
            )
            
            assert isinstance(prompt, str)
            assert len(prompt) > 100  # Should be substantial
            assert "test" in prompt.lower()
    
    def test_custom_to_template_equivalence(self):
        """Test that custom prompts match template behavior"""
        manager = PromptManager()
        
        schema = "Table: test [id, name]"
        question = "Count records"
        
        # Get few-shot template
        template_prompt = manager.get_prompt(
            prompt_type=PromptType.FEW_SHOT,
            schema=schema,
            question=question,
            max_examples=3
        )
        
        # Create equivalent custom prompt
        custom_prompt = manager.create_custom_prompt(
            schema=schema,
            question=question,
            include_examples=True,
            include_rules=False,
            include_patterns=False,
            max_examples=3
        )
        
        # Both should contain examples
        assert "Example 1:" in template_prompt
        assert "Example 1:" in custom_prompt
    
    def test_backward_compatibility(self):
        """Test backward compatibility with old TemplateManager name"""
        from src.models import TemplateManager
        
        manager = TemplateManager()
        assert isinstance(manager, PromptManager)
        
        prompt = manager.get_prompt(
            prompt_type=PromptType.BASIC,
            schema="Table: test [id]",
            question="Test"
        )
        
        assert isinstance(prompt, str)


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_schema(self):
        """Test with empty schema"""
        manager = PromptManager()
        
        prompt = manager.get_prompt(
            prompt_type=PromptType.BASIC,
            schema="",
            question="Test question"
        )
        
        assert isinstance(prompt, str)
        assert "Test question" in prompt
    
    def test_empty_question(self):
        """Test with empty question"""
        manager = PromptManager()
        
        prompt = manager.get_prompt(
            prompt_type=PromptType.BASIC,
            schema="Table: test [id]",
            question=""
        )
        
        assert isinstance(prompt, str)
        assert "Table: test" in prompt
    
    def test_very_long_schema(self):
        """Test with very long schema"""
        manager = PromptManager()
        
        long_schema = "\n".join([
            f"Table: table{i} [col1, col2, col3, col4, col5]"
            for i in range(50)
        ])
        
        prompt = manager.get_prompt(
            prompt_type=PromptType.BASIC,
            schema=long_schema,
            question="Test"
        )
        
        assert isinstance(prompt, str)
        assert "table0" in prompt
        assert "table49" in prompt
    
    def test_special_characters_in_question(self):
        """Test with special characters in question"""
        manager = PromptManager()
        
        question = "Show records where name='O'Brien' & salary>$50,000"
        
        prompt = manager.get_prompt(
            prompt_type=PromptType.BASIC,
            schema="Table: test [id]",
            question=question
        )
        
        assert question in prompt
    
    def test_unicode_in_inputs(self):
        """Test with unicode characters"""
        manager = PromptManager()
        
        prompt = manager.get_prompt(
            prompt_type=PromptType.BASIC,
            schema="Table: employés [id, nom, prénom]",
            question="Afficher tous les employés"
        )
        
        assert "employés" in prompt
        assert "Afficher" in prompt


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])