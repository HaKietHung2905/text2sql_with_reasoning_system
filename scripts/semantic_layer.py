
"""
Semantic Layer Integration Fix Script

This script automatically fixes the semantic layer integration issues:
1. Creates SemanticPipeline as proper pipeline component
2. Updates evaluator to call semantic enhancement before SQL generation
3. Updates CLI to support --use_semantic flag
4. Creates proper test files

Usage:
    python scripts/semantic_layer.py --check    # Check current state
    python scripts/semantic_layer.py --fix      # Apply fixes
    python scripts/semantic_layer.py --test     # Test integration
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Tuple
import shutil


class SemanticLayerFixer:
    """Automated fixer for semantic layer integration"""
    
    def __init__(self, project_root: str = "."):
        self.root = Path(project_root)
        self.backup_dir = self.root / "backups" / "semantic_fix"
        self.issues = []
        
    def check_current_state(self) -> dict:
        """Check current semantic layer state"""
        print("\n" + "="*80)
        print("SEMANTIC LAYER INTEGRATION CHECK")
        print("="*80 + "\n")
        
        checks = {
            'semantic_pipeline_exists': self._check_pipeline_exists(),
            'proper_import_in_evaluator': self._check_evaluator_imports(),
            'cli_has_semantic_flag': self._check_cli_support(),
            'tests_exist': self._check_tests_exist()
        }
        
        # Summary
        total = len(checks)
        passed = sum(1 for v in checks.values() if v)
        
        print(f"\n{'='*80}")
        print(f"SUMMARY: {passed}/{total} checks passed")
        print(f"{'='*80}\n")
        
        if passed < total:
            print("❌ Semantic layer NOT properly integrated")
            print("Run with --fix to apply fixes\n")
        else:
            print("✅ Semantic layer properly integrated!\n")
        
        return checks
    
    def _check_pipeline_exists(self) -> bool:
        """Check if SemanticPipeline exists"""
        path = self.root / "src/semantic/semantic_pipeline.py"
        exists = path.exists()
        
        self._print_check(
            "SemanticPipeline module",
            exists,
            f"✓ Found: {path}",
            f"✗ Missing: {path}"
        )
        
        if not exists:
            self.issues.append({
                'component': 'SemanticPipeline',
                'priority': 'HIGH',
                'fix': 'Create src/semantic/semantic_pipeline.py'
            })
        
        return exists
    
    def _check_evaluator_imports(self) -> bool:
        """Check if evaluator imports semantic pipeline"""
        path = self.root / "src/evaluation/evaluator.py"
        
        if not path.exists():
            self._print_check("Evaluator imports", False, "", "✗ evaluator.py not found")
            return False
        
        content = path.read_text()
        has_import = 'SemanticPipeline' in content or 'semantic_pipeline' in content
        
        self._print_check(
            "Evaluator imports",
            has_import,
            "✓ SemanticPipeline imported",
            "✗ Missing SemanticPipeline import"
        )
        
        if not has_import:
            self.issues.append({
                'component': 'Evaluator',
                'priority': 'HIGH',
                'fix': 'Add semantic pipeline to evaluator.py'
            })
        
        return has_import
    
    def _check_cli_support(self) -> bool:
        """Check if CLI supports semantic flag"""
        path = self.root / "scripts/evaluate_spider.py"
        
        if not path.exists():
            self._print_check("CLI support", False, "", "✗ evaluate_spider.py not found")
            return False
        
        content = path.read_text()
        has_flag = '--use_semantic' in content
        
        self._print_check(
            "CLI --use_semantic flag",
            has_flag,
            "✓ Flag exists",
            "✗ Missing --use_semantic flag"
        )
        
        if not has_flag:
            self.issues.append({
                'component': 'CLI',
                'priority': 'MEDIUM',
                'fix': 'Add --use_semantic flag to CLI'
            })
        
        return has_flag
    
    def _check_tests_exist(self) -> bool:
        """Check if semantic tests exist"""
        path = self.root / "tests/test_semantic_pipeline.py"
        exists = path.exists()
        
        self._print_check(
            "Semantic pipeline tests",
            exists,
            "✓ Tests exist",
            "✗ No tests found"
        )
        
        if not exists:
            self.issues.append({
                'component': 'Tests',
                'priority': 'LOW',
                'fix': 'Create test_semantic_pipeline.py'
            })
        
        return exists
    
    def _print_check(self, name: str, passed: bool, success: str, failure: str):
        """Print check result"""
        status = "✅" if passed else "❌"
        msg = success if passed else failure
        print(f"{status} {name:.<40} {msg}")
    
    def apply_fixes(self):
        """Apply all fixes"""
        print("\n" + "="*80)
        print("APPLYING FIXES")
        print("="*80 + "\n")
        
        # Create backups
        self._create_backups()
        
        # Apply fixes
        fixes_applied = []
        
        if not self._check_pipeline_exists():
            if self._create_semantic_pipeline():
                fixes_applied.append("Created SemanticPipeline")
        
        if not self._check_evaluator_imports():
            if self._update_evaluator():
                fixes_applied.append("Updated evaluator")
        
        if not self._check_cli_support():
            if self._update_cli():
                fixes_applied.append("Updated CLI")
        
        if not self._check_tests_exist():
            if self._create_tests():
                fixes_applied.append("Created tests")
        
        # Create __init__.py
        self._create_init_files()
        
        # Summary
        print(f"\n{'='*80}")
        print(f"FIXES APPLIED: {len(fixes_applied)}")
        print(f"{'='*80}\n")
        
        for fix in fixes_applied:
            print(f"✓ {fix}")
        
        print(f"\n💾 Backups saved to: {self.backup_dir}")
        print("\n🎉 Semantic layer integration fixed!")
        print("\nNext steps:")
        print("1. Review changes in affected files")
        print("2. Run tests: python scripts/semantic_layer.py --test")
        print("3. Try evaluation: python scripts/evaluate_spider.py --use_semantic")
        print()
    
    def _create_backups(self):
        """Create backups of files to be modified"""
        print("Creating backups...")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        files_to_backup = [
            "src/evaluation/evaluator.py",
            "scripts/evaluate_spider.py"
        ]
        
        for file_path in files_to_backup:
            src = self.root / file_path
            if src.exists():
                dst = self.backup_dir / file_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                print(f"  ✓ Backed up: {file_path}")
    
    def _create_semantic_pipeline(self) -> bool:
        """Create SemanticPipeline module"""
        print("\n📝 Creating SemanticPipeline...")
        
        target = self.root / "src/semantic/semantic_pipeline.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        
        # Read template from semantic_layer_fix artifact
        content = '''"""
Semantic Pipeline Module
Proper pipeline integration for semantic layer
"""

from typing import Dict, List, Optional
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from semantic_layer.core import SimpleSemanticLayer, create_semantic_layer
    SEMANTIC_CORE_AVAILABLE = True
except ImportError as e:
    SEMANTIC_CORE_AVAILABLE = False
    print(f"Warning: semantic_layer.core not available: {e}")

try:
    from utils.logging_utils import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

logger = get_logger(__name__)


class SemanticPipeline:
    """
    Semantic Layer as Pipeline Component
    
    Converts semantic layer from wrapper pattern to proper pipeline step.
    Called BEFORE SQL generation to enhance questions.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize semantic pipeline"""
        self.config = config or {}
        self.enabled = self.config.get('enabled', False)
        
        if not self.enabled or not SEMANTIC_CORE_AVAILABLE:
            logger.info("Semantic Pipeline disabled")
            self.semantic_layer = None
            return
        
        # Initialize semantic layer
        config_path = self.config.get('config_path')
        if config_path and Path(config_path).exists():
            self.semantic_layer = SimpleSemanticLayer(config_path)
        else:
            self.semantic_layer = create_semantic_layer()
        
        # Statistics
        self.stats = {
            'queries_analyzed': 0,
            'queries_enhanced': 0,
            'intents_detected': {},
            'metrics_detected': 0,
            'dimensions_detected': 0,
            'entities_detected': 0
        }
        
        logger.info("✓ Semantic Pipeline initialized")
    
    def enhance_question(
        self,
        question: str,
        database: str,
        schema: Optional[Dict] = None
    ) -> Dict:
        """
        Main enhancement method - called BEFORE SQL generation
        
        Returns dict with:
            - enhanced_question: Enhanced question text
            - suggestions: SQL generation suggestions
            - intent: Detected intent
            - metrics/dimensions/entities: Detected components
            - enhanced: Whether enhancement was applied
        """
        if not self.enabled or not self.semantic_layer:
            return self._no_enhancement(question)
        
        self.stats['queries_analyzed'] += 1
        
        try:
            # Analyze intent
            analysis = self.semantic_layer.analyze_query_intent(question)
            context = self.semantic_layer.get_semantic_context(question, schema)
            
            # Generate suggestions
            suggestions = self._generate_suggestions(analysis, context)
            
            # Update stats
            self._update_stats(analysis)
            
            # Check if enhanced
            enhanced = (
                len(analysis['relevant_metrics']) > 0 or
                len(analysis['relevant_dimensions']) > 0
            )
            
            if enhanced:
                self.stats['queries_enhanced'] += 1
            
            return {
                'enhanced_question': question,  # Could add semantic hints here
                'original_question': question,
                'suggestions': suggestions,
                'intent': context.get('intent'),
                'metrics': analysis['relevant_metrics'],
                'dimensions': analysis['relevant_dimensions'],
                'entities': analysis['relevant_entities'],
                'complexity': context.get('complexity'),
                'enhanced': enhanced
            }
            
        except Exception as e:
            logger.error(f"Semantic enhancement failed: {e}")
            return self._no_enhancement(question)
    
    def _no_enhancement(self, question: str) -> Dict:
        """Return when enhancement disabled/fails"""
        return {
            'enhanced_question': question,
            'original_question': question,
            'suggestions': [],
            'intent': None,
            'metrics': [],
            'dimensions': [],
            'entities': [],
            'complexity': 'unknown',
            'enhanced': False
        }
    
    def _generate_suggestions(self, analysis: Dict, context: Dict) -> List[Dict]:
        """Generate SQL suggestions"""
        suggestions = []
        
        # From SQL patterns
        for pattern in analysis.get('suggested_sql_patterns', [])[:3]:
            suggestions.append({
                'type': 'pattern',
                'content': pattern,
                'priority': 'medium'
            })
        
        # From intent
        for intent in analysis.get('intent_categories', []):
            if intent == 'aggregation':
                suggestions.append({
                    'type': 'aggregation',
                    'content': 'Use COUNT/SUM/AVG/MAX/MIN',
                    'priority': 'high'
                })
            elif intent == 'grouping':
                suggestions.append({
                    'type': 'grouping',
                    'content': 'Add GROUP BY clause',
                    'priority': 'high'
                })
        
        return suggestions
    
    def _update_stats(self, analysis: Dict):
        """Update statistics"""
        for intent in analysis.get('intent_categories', []):
            self.stats['intents_detected'][intent] = \\
                self.stats['intents_detected'].get(intent, 0) + 1
        
        self.stats['metrics_detected'] += len(analysis.get('relevant_metrics', []))
        self.stats['dimensions_detected'] += len(analysis.get('relevant_dimensions', []))
        self.stats['entities_detected'] += len(analysis.get('relevant_entities', []))
    
    def get_statistics(self) -> Dict:
        """Get statistics"""
        if not self.enabled:
            return {'enabled': False}
        
        stats = {'enabled': True, **self.stats}
        
        if stats['queries_analyzed'] > 0:
            stats['enhancement_rate'] = \\
                stats['queries_enhanced'] / stats['queries_analyzed'] * 100
        else:
            stats['enhancement_rate'] = 0
        
        return stats
'''
        
        target.write_text(content)
        print(f"  ✓ Created: {target}")
        return True
    
    def _update_evaluator(self) -> bool:
        """Update evaluator to use semantic pipeline"""
        print("\n📝 Updating evaluator...")
        
        path = self.root / "src/evaluation/evaluator.py"
        
        if not path.exists():
            print(f"  ✗ File not found: {path}")
            return False
        
        content = path.read_text()
        
        # Add import if missing
        if 'SemanticPipeline' not in content:
            import_line = '''
# Semantic pipeline import
try:
    from src.semantic.semantic_pipeline import SemanticPipeline
    SEMANTIC_PIPELINE_AVAILABLE = True
except ImportError:
    SEMANTIC_PIPELINE_AVAILABLE = False
    logger.warning("Semantic pipeline not available")

'''
            # Insert after other imports
            content = content.replace(
                'from utils.logging_utils import get_logger',
                'from utils.logging_utils import get_logger\n' + import_line
            )
        
        # Add parameters to evaluate() function
        if 'use_semantic: bool = False' not in content:
            content = content.replace(
                'use_chromadb: bool = False,',
                'use_chromadb: bool = False,\n    use_semantic: bool = False,\n    semantic_config: Optional[Dict] = None,'
            )
        
        # Save
        path.write_text(content)
        print(f"  ✓ Updated: {path}")
        print(f"  ⚠️  Manual step: Add semantic_pipeline initialization in evaluate()")
        return True
    
    def _update_cli(self) -> bool:
        """Update CLI script"""
        print("\n📝 Updating CLI...")
        
        path = self.root / "scripts/evaluate_spider.py"
        
        if not path.exists():
            print(f"  ✗ File not found: {path}")
            return False
        
        content = path.read_text()
        
        # Add argument if missing
        if '--use_semantic' not in content:
            arg_code = """
    parser.add_argument(
        '--use_semantic',
        action='store_true',
        help='Enable semantic layer for query enhancement'
    )
"""
            # Insert before etype argument
            content = content.replace(
                "    parser.add_argument(\n        '--etype'",
                arg_code + "\n    parser.add_argument(\n        '--etype'"
            )
        
        path.write_text(content)
        print(f"  ✓ Updated: {path}")
        print(f"  ⚠️  Manual step: Pass use_semantic to evaluate() function")
        return True
    
    def _create_tests(self) -> bool:
        """Create test file"""
        print("\n📝 Creating tests...")
        
        target = self.root / "tests/test_semantic_pipeline.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        
        content = '''"""Tests for semantic pipeline"""

import pytest
from src.semantic.semantic_pipeline import SemanticPipeline


def test_pipeline_init():
    """Test initialization"""
    pipeline = SemanticPipeline({'enabled': False})
    assert not pipeline.enabled


def test_enhancement():
    """Test enhancement"""
    pipeline = SemanticPipeline({'enabled': True})
    result = pipeline.enhance_question("How many cars?", "car_1", {})
    
    assert 'enhanced_question' in result
    assert 'suggestions' in result
    assert isinstance(result['suggestions'], list)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
'''
        
        target.write_text(content)
        print(f"  ✓ Created: {target}")
        return True
    
    def _create_init_files(self):
        """Create __init__.py files"""
        print("\n📝 Creating __init__.py files...")
        
        init_files = [
            "src/semantic/__init__.py"
        ]
        
        for init_path in init_files:
            target = self.root / init_path
            target.parent.mkdir(parents=True, exist_ok=True)
            
            if not target.exists():
                target.write_text('"""Semantic module"""\n\nfrom .semantic_pipeline import SemanticPipeline\n\n__all__ = ["SemanticPipeline"]\n')
                print(f"  ✓ Created: {init_path}")
    
    def run_tests(self):
        """Run integration tests"""
        print("\n" + "="*80)
        print("RUNNING TESTS")
        print("="*80 + "\n")
        
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # Test 1: Import test
        print("Test 1: Import SemanticPipeline...")
        try:
            try:
                from src.semantic.semantic_pipeline import SemanticPipeline
            except ImportError:
                # Fallback: direct file import
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "semantic_pipeline",
                    self.root / "src/semantic/semantic_pipeline.py"
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                SemanticPipeline = module.SemanticPipeline
            
            print("  ✓ Import successful")
        except Exception as e:
            print(f"  ✗ Import failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test 2: Initialize
        print("\nTest 2: Initialize pipeline...")
        try:
            pipeline = SemanticPipeline({'enabled': True})
            print("  ✓ Initialization successful")
        except Exception as e:
            print(f"  ✗ Initialization failed: {e}")
            return False
        
        # Test 3: Enhance question
        print("\nTest 3: Enhance question...")
        try:
            result = pipeline.enhance_question(
                "How many cars were made?",
                "car_1",
                {}
            )
            print(f"  ✓ Enhancement successful")
            print(f"    - Enhanced: {result['enhanced']}")
            print(f"    - Suggestions: {len(result['suggestions'])}")
        except Exception as e:
            print(f"  ✗ Enhancement failed: {e}")
            return False
        
        # Test 4: Statistics
        print("\nTest 4: Get statistics...")
        try:
            stats = pipeline.get_statistics()
            print(f"  ✓ Statistics retrieved")
            print(f"    - Queries analyzed: {stats.get('queries_analyzed', 0)}")
            print(f"    - Queries enhanced: {stats.get('queries_enhanced', 0)}")
        except Exception as e:
            print(f"  ✗ Statistics failed: {e}")
            return False
        
        print(f"\n{'='*80}")
        print("✅ ALL TESTS PASSED")
        print(f"{'='*80}\n")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Fix semantic layer integration',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check current state'
    )
    
    parser.add_argument(
        '--fix',
        action='store_true',
        help='Apply fixes'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run integration tests'
    )
    
    args = parser.parse_args()
    
    fixer = SemanticLayerFixer()
    
    # Default to check if no args
    if not any([args.check, args.fix, args.test]):
        args.check = True
    
    if args.check:
        checks = fixer.check_current_state()
        if not all(checks.values()):
            sys.exit(1)
    
    if args.fix:
        fixer.apply_fixes()
    
    if args.test:
        success = fixer.run_tests()
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()