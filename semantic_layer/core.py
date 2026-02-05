"""
Semantic Layer Core Module
==========================

Core implementation of the semantic layer for Text-to-SQL enhancement.
This module provides semantic understanding capabilities to improve SQL generation.

Classes:
    - SimpleSemanticLayer: Main semantic layer implementation
    - Metric: Business metric definition
    - Dimension: Data dimension definition  
    - Entity: Business entity definition
    - MetricType, DimensionType: Enumeration types

Functions:
    - create_semantic_layer: Factory function to create configured semantic layer
    - enhance_sql_generation: Simple function to enhance SQL with semantics
"""

import json
import re
import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

class MetricType(Enum):
    """Types of metrics supported by the semantic layer"""
    COUNT = "count"
    DISTINCT_COUNT = "distinct_count"
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    PERCENTAGE = "percentage"
    RATIO = "ratio"

class DimensionType(Enum):
    """Types of dimensions in the semantic layer"""
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    NUMERICAL = "numerical"
    GEOGRAPHICAL = "geographical"
    BOOLEAN = "boolean"

@dataclass
class Metric:
    """Represents a business metric in the semantic layer"""
    name: str
    description: str
    type: MetricType
    sql_expression: str
    base_table: str
    keywords: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    filters: List[str] = field(default_factory=list)
    
    def matches_query(self, query: str) -> bool:
        """Check if this metric is relevant to the given query"""
        query_lower = query.lower()
        return any(keyword.lower() in query_lower for keyword in self.keywords)
    
    def to_dict(self) -> Dict:
        """Convert metric to dictionary representation"""
        return {
            'name': self.name,
            'description': self.description,
            'type': self.type.value,
            'sql_expression': self.sql_expression,
            'base_table': self.base_table,
            'keywords': self.keywords,
            'dependencies': self.dependencies,
            'filters': self.filters
        }

@dataclass
class Dimension:
    """Represents a dimension in the semantic layer"""
    name: str
    description: str
    type: DimensionType
    column: str
    table: str
    keywords: List[str] = field(default_factory=list)
    hierarchy: Optional[List[str]] = None
    aliases: List[str] = field(default_factory=list)
    
    def matches_query(self, query: str) -> bool:
        """Check if this dimension is relevant to the given query"""
        query_lower = query.lower()
        all_keywords = self.keywords + self.aliases + [self.name.lower()]
        return any(keyword.lower() in query_lower for keyword in all_keywords)
    
    def to_dict(self) -> Dict:
        """Convert dimension to dictionary representation"""
        return {
            'name': self.name,
            'description': self.description,
            'type': self.type.value,
            'column': self.column,
            'table': self.table,
            'keywords': self.keywords,
            'hierarchy': self.hierarchy,
            'aliases': self.aliases
        }

@dataclass
class Entity:
    """Represents a business entity in the semantic layer"""
    name: str
    primary_table: str
    keywords: List[str] = field(default_factory=list)
    related_tables: List[str] = field(default_factory=list)
    common_columns: Dict[str, List[str]] = field(default_factory=dict)
    relationships: List[Dict] = field(default_factory=list)
    
    def matches_query(self, query: str) -> bool:
        """Check if this entity is relevant to the given query"""
        query_lower = query.lower()
        return any(keyword.lower() in query_lower for keyword in self.keywords)
    
    def to_dict(self) -> Dict:
        """Convert entity to dictionary representation"""
        return {
            'name': self.name,
            'primary_table': self.primary_table,
            'keywords': self.keywords,
            'related_tables': self.related_tables,
            'common_columns': self.common_columns,
            'relationships': self.relationships
        }

class SimpleSemanticLayer:
    """
    Main semantic layer class that provides business logic and semantic understanding
    for SQL generation and data analysis.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.metrics: Dict[str, Metric] = {}
        self.dimensions: Dict[str, Dimension] = {}
        self.entities: Dict[str, Entity] = {}
        self.config: Dict = {}
        
        # Load configuration
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        else:
            # Try to load from default location
            default_config = Path(__file__).parent / "config.json"
            if default_config.exists():
                self.load_config(str(default_config))
            else:
                # Initialize with default semantic objects
                self._setup_default_semantic_objects()
    
    def load_config(self, config_path: str):
        """Load semantic layer configuration from JSON file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                self.config = config_data.get('semantic_layer_config', config_data)
            
            # Setup semantic objects from config
            self._setup_from_config()
            
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            self._setup_default_semantic_objects()
    
    def _setup_from_config(self):
        """Setup semantic objects from loaded configuration"""
        
        # Setup metrics from config patterns
        metrics_config = self.config.get('metrics', {})
        
        # Count metrics
        count_patterns = metrics_config.get('count_patterns', [])
        if count_patterns:
            self.add_metric(Metric(
                name="record_count",
                description="Count of records",
                type=MetricType.COUNT,
                sql_expression="COUNT(*)",
                base_table="main",
                keywords=count_patterns
            ))
        
        # Distinct count metrics
        distinct_patterns = metrics_config.get('distinct_count_patterns', [])
        if distinct_patterns:
            self.add_metric(Metric(
                name="distinct_count",
                description="Count of distinct values",
                type=MetricType.DISTINCT_COUNT,
                sql_expression="COUNT(DISTINCT {column})",
                base_table="main",
                keywords=distinct_patterns
            ))
        
        # Average metrics
        avg_patterns = metrics_config.get('average_patterns', [])
        if avg_patterns:
            self.add_metric(Metric(
                name="average_value",
                description="Average value",
                type=MetricType.AVERAGE,
                sql_expression="AVG({column})",
                base_table="main",
                keywords=avg_patterns
            ))
        
        # Sum metrics
        sum_patterns = metrics_config.get('sum_patterns', [])
        if sum_patterns:
            self.add_metric(Metric(
                name="sum_value",
                description="Sum of values",
                type=MetricType.SUM,
                sql_expression="SUM({column})",
                base_table="main",
                keywords=sum_patterns
            ))
        
        # Max metrics
        max_patterns = metrics_config.get('max_patterns', [])
        if max_patterns:
            self.add_metric(Metric(
                name="max_value",
                description="Maximum value",
                type=MetricType.MAX,
                sql_expression="MAX({column})",
                base_table="main",
                keywords=max_patterns
            ))
        
        # Min metrics
        min_patterns = metrics_config.get('min_patterns', [])
        if min_patterns:
            self.add_metric(Metric(
                name="min_value",
                description="Minimum value",
                type=MetricType.MIN,
                sql_expression="MIN({column})",
                base_table="main",
                keywords=min_patterns
            ))
        
        # Setup dimensions from config
        dimensions_config = self.config.get('dimensions', {})
        
        # Temporal dimensions
        temporal_patterns = dimensions_config.get('temporal_patterns', [])
        if temporal_patterns:
            self.add_dimension(Dimension(
                name="temporal_dimension",
                description="Time-based grouping",
                type=DimensionType.TEMPORAL,
                column="date",
                table="main",
                keywords=temporal_patterns
            ))
        
        # Categorical dimensions
        categorical_patterns = dimensions_config.get('categorical_patterns', [])
        if categorical_patterns:
            self.add_dimension(Dimension(
                name="categorical_dimension",
                description="Category-based grouping",
                type=DimensionType.CATEGORICAL,
                column="category",
                table="main",
                keywords=categorical_patterns
            ))
        
        # Setup entities from config
        entities_config = self.config.get('entities', {})
        for entity_name, entity_data in entities_config.items():
            if isinstance(entity_data, dict):
                entity = Entity(
                    name=entity_name,
                    primary_table=entity_data.get('primary_table', ''),
                    keywords=entity_data.get('keywords', []),
                    related_tables=entity_data.get('related_tables', []),
                    common_columns=entity_data.get('common_columns', {}),
                    relationships=entity_data.get('relationships', [])
                )
                self.add_entity(entity)
    
    def _setup_default_semantic_objects(self):
        """Setup default metrics, dimensions, and entities"""
        
        # Default metrics
        default_metrics = [
            Metric("record_count", "Count of records", MetricType.COUNT, "COUNT(*)", "main",
                   ["how many", "count", "number of", "total"]),
            Metric("distinct_count", "Count of distinct values", MetricType.DISTINCT_COUNT, "COUNT(DISTINCT {column})", "main",
                   ["unique", "distinct", "different"]),
            Metric("average_value", "Average value", MetricType.AVERAGE, "AVG({column})", "main",
                   ["average", "mean", "avg"]),
            Metric("sum_value", "Sum of values", MetricType.SUM, "SUM({column})", "main",
                   ["total", "sum", "amount"]),
            Metric("max_value", "Maximum value", MetricType.MAX, "MAX({column})", "main",
                   ["maximum", "max", "highest", "largest", "top"]),
            Metric("min_value", "Minimum value", MetricType.MIN, "MIN({column})", "main",
                   ["minimum", "min", "lowest", "smallest", "bottom"])
        ]
        
        for metric in default_metrics:
            self.add_metric(metric)
        
        # Default dimensions
        default_dimensions = [
            Dimension("temporal_dimension", "Time-based grouping", DimensionType.TEMPORAL, "date", "main",
                     ["by year", "by month", "by day", "over time", "yearly", "monthly", "daily"]),
            Dimension("categorical_dimension", "Category-based grouping", DimensionType.CATEGORICAL, "category", "main",
                     ["by category", "by type", "by group", "by class"])
        ]
        
        for dimension in default_dimensions:
            self.add_dimension(dimension)
        
        # Default entities (Spider dataset specific)
        default_entities = [
            Entity("car", "cars_data", ["car", "vehicle", "automobile", "auto"], 
                   ["car_names", "car_makers", "model_list"]),
            Entity("student", "students", ["student", "pupil", "learner"], 
                   ["enrollments", "grades", "courses"]),
            Entity("customer", "customers", ["customer", "client", "buyer"], 
                   ["orders", "payments"])
        ]
        
        for entity in default_entities:
            self.add_entity(entity)
    
    def add_metric(self, metric: Metric):
        """Add a metric to the semantic layer"""
        self.metrics[metric.name] = metric
    
    def add_dimension(self, dimension: Dimension):
        """Add a dimension to the semantic layer"""
        self.dimensions[dimension.name] = dimension
    
    def add_entity(self, entity: Entity):
        """Add an entity to the semantic layer"""
        self.entities[entity.name] = entity
    
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze query to understand intent and suggest enhancements"""
        query_lower = query.lower()
        
        analysis = {
            'query': query,
            'relevant_metrics': [],
            'relevant_dimensions': [],
            'relevant_entities': [],
            'suggested_sql_patterns': [],
            'complexity_score': 0,
            'intent_categories': []
        }
        
        # Find relevant metrics
        for metric in self.metrics.values():
            if metric.matches_query(query):
                analysis['relevant_metrics'].append(metric.to_dict())
                analysis['complexity_score'] += 2
        
        # Find relevant dimensions
        for dimension in self.dimensions.values():
            if dimension.matches_query(query):
                analysis['relevant_dimensions'].append(dimension.to_dict())
                analysis['complexity_score'] += 1
        
        # Find relevant entities
        for entity in self.entities.values():
            if entity.matches_query(query):
                analysis['relevant_entities'].append(entity.to_dict())
                analysis['complexity_score'] += 1
        
        # Analyze intent categories
        analysis['intent_categories'] = self._analyze_intent_categories(query)
        
        # Generate SQL pattern suggestions
        analysis['suggested_sql_patterns'] = self._generate_sql_suggestions(analysis)
        
        return analysis
    
    def _analyze_intent_categories(self, query: str) -> List[str]:
        """Analyze the intent categories present in the query"""
        query_lower = query.lower()
        categories = []
        
        # Check for aggregation intent
        if any(word in query_lower for word in ['count', 'total', 'sum', 'average', 'max', 'min']):
            categories.append('aggregation')
        
        # Check for grouping intent
        if any(word in query_lower for word in ['by', 'per', 'group', 'each', 'every']):
            categories.append('grouping')
        
        # Check for filtering intent
        if any(word in query_lower for word in ['where', 'with', 'having', 'that', 'which']):
            categories.append('filtering')
        
        # Check for ordering intent
        if any(word in query_lower for word in ['top', 'bottom', 'highest', 'lowest', 'first', 'last']):
            categories.append('ordering')
        
        # Check for comparison intent
        if any(word in query_lower for word in ['greater', 'less', 'more', 'fewer', 'above', 'below']):
            categories.append('comparison')
        
        return categories
    
    def _generate_sql_suggestions(self, analysis: Dict) -> List[str]:
        """Generate SQL pattern suggestions based on analysis"""
        suggestions = []
        
        metrics = analysis['relevant_metrics']
        dimensions = analysis['relevant_dimensions']
        entities = analysis['relevant_entities']
        intent_categories = analysis['intent_categories']
        
        # Basic metric patterns
        for metric in metrics:
            metric_type = metric['type']
            if metric_type == 'count':
                suggestions.append("SELECT COUNT(*) FROM {table}")
            elif metric_type == 'average':
                suggestions.append("SELECT AVG({column}) FROM {table}")
            elif metric_type == 'sum':
                suggestions.append("SELECT SUM({column}) FROM {table}")
            elif metric_type == 'max':
                suggestions.append("SELECT MAX({column}) FROM {table}")
            elif metric_type == 'min':
                suggestions.append("SELECT MIN({column}) FROM {table}")
        
        # Grouping patterns
        if dimensions and 'grouping' in intent_categories:
            suggestions.append("SELECT {metric}, {dimension} FROM {table} GROUP BY {dimension}")
        
        # Ordering patterns
        if 'ordering' in intent_categories:
            suggestions.append("SELECT * FROM {table} ORDER BY {column} DESC")
            suggestions.append("SELECT * FROM {table} ORDER BY {column} ASC")
        
        # Filtering patterns
        if 'filtering' in intent_categories or 'comparison' in intent_categories:
            suggestions.append("SELECT * FROM {table} WHERE {column} > {value}")
            suggestions.append("SELECT * FROM {table} WHERE {column} = '{value}'")
        
        # Join patterns for multiple entities
        if len(entities) > 1:
            suggestions.append("Consider JOINs between related tables")
            suggestions.append("SELECT * FROM {table1} JOIN {table2} ON {condition}")
        
        # Complex patterns
        if len(metrics) > 0 and len(dimensions) > 0:
            suggestions.append("SELECT {group_column}, {aggregation} FROM {table} GROUP BY {group_column}")
        
        return list(set(suggestions))  # Remove duplicates
    
    def enhance_sql_with_semantics(self, base_sql: str, query: str) -> str:
        """Enhance existing SQL with semantic understanding"""
        analysis = self.analyze_query_intent(query)
        enhanced_sql = base_sql
        
        # Apply enhancements based on analysis
        enhanced_sql = self._apply_metric_enhancements(enhanced_sql, analysis)
        enhanced_sql = self._apply_dimension_enhancements(enhanced_sql, analysis)
        enhanced_sql = self._apply_entity_enhancements(enhanced_sql, analysis)
        
        return enhanced_sql
    
    def _apply_metric_enhancements(self, sql: str, analysis: Dict) -> str:
        """Apply metric-based enhancements to SQL"""
        relevant_metrics = analysis['relevant_metrics']
        
        if not relevant_metrics:
            return sql
        
        # Simple enhancement: replace SELECT * with appropriate aggregation
        for metric in relevant_metrics:
            metric_type = metric['type']
            
            if 'SELECT *' in sql.upper():
                if metric_type == 'count':
                    sql = sql.replace('SELECT *', 'SELECT COUNT(*)', 1)
                    break
                elif metric_type in ['average', 'sum', 'max', 'min']:
                    # Would need column detection here in real implementation
                    sql = sql + f" -- Consider: {metric['sql_expression']}"
                    break
        
        return sql
    
    def _apply_dimension_enhancements(self, sql: str, analysis: Dict) -> str:
        """Apply dimension-based enhancements to SQL"""
        relevant_dimensions = analysis['relevant_dimensions']
        
        if not relevant_dimensions or 'GROUP BY' in sql.upper():
            return sql
        
        # Add GROUP BY suggestion
        for dimension in relevant_dimensions:
            if dimension['type'] in ['categorical', 'temporal']:
                sql = sql + f" -- Consider: GROUP BY {dimension['column']}"
                break
        
        return sql
    
    def _apply_entity_enhancements(self, sql: str, analysis: Dict) -> str:
        """Apply entity-based enhancements to SQL"""
        relevant_entities = analysis['relevant_entities']
        
        if len(relevant_entities) <= 1:
            return sql
        
        # Suggest joins for multiple entities
        sql = sql + " -- Consider: JOINs between related tables"
        
        return sql
    
    def get_semantic_context(self, query: str, schema_info: Dict = None) -> Dict[str, Any]:
        """Get comprehensive semantic context for a query"""
        analysis = self.analyze_query_intent(query)
        
        context = {
            'semantic_analysis': analysis,
            'suggestions': {
                'metrics': analysis['relevant_metrics'],
                'dimensions': analysis['relevant_dimensions'],
                'entities': analysis['relevant_entities'],
                'sql_patterns': analysis['suggested_sql_patterns']
            },
            'complexity': self._assess_complexity(analysis),
            'recommended_approach': self._recommend_approach(analysis),
            'enhancement_opportunities': self._identify_enhancement_opportunities(analysis)
        }
        
        return context
    
    def _assess_complexity(self, analysis: Dict) -> str:
        """Assess query complexity based on semantic analysis"""
        score = analysis['complexity_score']
        
        if score <= 2:
            return "Simple - direct query"
        elif score <= 5:
            return "Medium - may need aggregation or grouping"
        elif score <= 8:
            return "Complex - likely needs joins and multiple operations"
        else:
            return "Very Complex - advanced query with multiple features"
    
    def _recommend_approach(self, analysis: Dict) -> str:
        """Recommend SQL approach based on analysis"""
        metrics = analysis['relevant_metrics']
        dimensions = analysis['relevant_dimensions']
        entities = analysis['relevant_entities']
        intent_categories = analysis['intent_categories']
        
        if not metrics and not dimensions:
            return "Simple SELECT with optional WHERE clause"
        elif metrics and not dimensions:
            return "Aggregation query without grouping"
        elif metrics and dimensions:
            return "Aggregation with GROUP BY clause"
        elif 'ordering' in intent_categories:
            return "Query with ORDER BY clause"
        elif len(entities) > 1:
            return "Multi-table query with JOINs"
        else:
            return "Standard SELECT query with filtering"
    
    def _identify_enhancement_opportunities(self, analysis: Dict) -> List[str]:
        """Identify specific enhancement opportunities"""
        opportunities = []
        
        metrics = analysis['relevant_metrics']
        dimensions = analysis['relevant_dimensions']
        entities = analysis['relevant_entities']
        intent_categories = analysis['intent_categories']
        
        if metrics:
            opportunities.append("Add appropriate aggregation functions")
        
        if dimensions:
            opportunities.append("Consider GROUP BY for dimensional analysis")
        
        if 'ordering' in intent_categories:
            opportunities.append("Add ORDER BY for sorted results")
        
        if len(entities) > 1:
            opportunities.append("Join related tables for comprehensive data")
        
        if 'filtering' in intent_categories:
            opportunities.append("Add WHERE clause for data filtering")
        
        return opportunities
    
    def find_relevant_columns(self, query: str, schema_info: Dict) -> Dict[str, List[str]]:
        """Find relevant columns based on query intent and schema"""
        column_hints = self.config.get('column_type_hints', {})
        
        relevant_columns = {
            'numeric': [],
            'categorical': [],
            'temporal': [],
            'textual': []
        }
        
        if not schema_info:
            return relevant_columns
        
        query_lower = query.lower()
        
        # Check all tables and columns
        for table, columns in schema_info.items():
            for column in columns:
                column_lower = column.lower()
                
                # Check if column matches query terms
                if any(term in column_lower for term in query_lower.split()):
                    # Categorize column based on name hints
                    if any(hint in column_lower for hint in column_hints.get('numeric_indicators', [])):
                        relevant_columns['numeric'].append(f"{table}.{column}")
                    elif any(hint in column_lower for hint in column_hints.get('categorical_indicators', [])):
                        relevant_columns['categorical'].append(f"{table}.{column}")
                    elif any(hint in column_lower for hint in column_hints.get('temporal_indicators', [])):
                        relevant_columns['temporal'].append(f"{table}.{column}")
                    elif any(hint in column_lower for hint in column_hints.get('textual_indicators', [])):
                        relevant_columns['textual'].append(f"{table}.{column}")
        
        return relevant_columns

# Factory function to create and configure semantic layer
def create_semantic_layer(config_path: Optional[str] = None) -> SimpleSemanticLayer:
    """
    Factory function to create a configured semantic layer
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Configured SimpleSemanticLayer instance
    """
    return SimpleSemanticLayer(config_path)

# Integration function for existing evaluation code
def enhance_sql_generation(question: str, base_sql: str = "", schema_info: Dict = None) -> Dict[str, Any]:
    """
    Simple function to enhance SQL generation with semantic understanding
    Can be easily integrated into existing evaluation code
    
    Args:
        question: Natural language question
        base_sql: Base SQL query to enhance (optional)
        schema_info: Database schema information (optional)
        
    Returns:
        Dictionary with enhancement results
    """
    semantic_layer = create_semantic_layer()
    
    # Get semantic analysis
    context = semantic_layer.get_semantic_context(question, schema_info)
    
    # Enhance SQL if provided
    enhanced_sql = base_sql
    if base_sql:
        enhanced_sql = semantic_layer.enhance_sql_with_semantics(base_sql, question)
    
    return {
        'original_sql': base_sql,
        'enhanced_sql': enhanced_sql,
        'semantic_context': context,
        'suggestions': context['suggestions'],
        'complexity': context['complexity'],
        'approach': context['recommended_approach'],
        'enhancement_opportunities': context['enhancement_opportunities']
    }

# Test and example functions
def test_semantic_layer():
    """Test the semantic layer with example queries"""
    
    print("🎯 Testing Semantic Layer")
    print("=" * 30)
    
    semantic_layer = create_semantic_layer()
    
    test_queries = [
        "How many cars were made in 1980?",
        "What is the average horsepower by manufacturer?",
        "Show me the top 10 cars with highest MPG",
        "List all students enrolled in computer science courses",
        "What's the total revenue by region over time?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        analysis = semantic_layer.analyze_query_intent(query)
        
        print(f"  Complexity: {semantic_layer._assess_complexity(analysis)}")
        print(f"  Approach: {semantic_layer._recommend_approach(analysis)}")
        print(f"  Metrics: {len(analysis['relevant_metrics'])}")
        print(f"  Dimensions: {len(analysis['relevant_dimensions'])}")
        print(f"  Entities: {len(analysis['relevant_entities'])}")
        
        # Show specific findings
        for metric in analysis['relevant_metrics']:
            print(f"    → Metric: {metric['name']} ({metric['type']})")
        
        for dimension in analysis['relevant_dimensions']:
            print(f"    → Dimension: {dimension['name']} ({dimension['type']})")
        
        for entity in analysis['relevant_entities']:
            print(f"    → Entity: {entity['name']}")

if __name__ == "__main__":
    # Run tests when module is executed directly
    test_semantic_layer()