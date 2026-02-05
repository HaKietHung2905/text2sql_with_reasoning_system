"""
Integration Test for Complete ReasoningBank Pipeline - FIXED VERSION

This script tests the end-to-end integration with proper error handling
"""

import sys
import os
from pathlib import Path
import json
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logging_utils import get_logger

logger = get_logger(__name__)


def test_imports():
    """Test that all required modules can be imported"""
    logger.info("="*80)
    logger.info("Testing Imports")
    logger.info("="*80)
    
    tests = []
    
    # Test semantic layer
    try:
        from src.semantic.semantic_pipeline import SemanticPipeline
        tests.append(("SemanticPipeline", True, None))
        logger.info("✓ SemanticPipeline imported")
    except Exception as e:
        tests.append(("SemanticPipeline", False, str(e)))
        logger.error(f"✗ SemanticPipeline import failed: {e}")
    
    # Test ReasoningBank components
    try:
        from src.reasoning import (
            ExperienceCollector,
            SelfJudgment,
            StrategyDistillation,
            ReasoningMemoryStore,
            MemoryRetrieval,
            MemoryConsolidation,
            ParallelScaling,
            ReasoningBankPipeline
        )
        tests.append(("ReasoningBank Components", True, None))
        logger.info("✓ ReasoningBank components imported")
    except Exception as e:
        tests.append(("ReasoningBank Components", False, str(e)))
        logger.error(f"✗ ReasoningBank import failed: {e}")
    
    # Test evaluator
    try:
        from src.evaluation.evaluator import evaluate
        tests.append(("Evaluator", True, None))
        logger.info("✓ Evaluator imported")
    except Exception as e:
        tests.append(("Evaluator", False, str(e)))
        logger.error(f"✗ Evaluator import failed: {e}")
    
    return tests


def test_semantic_pipeline():
    """Test semantic pipeline functionality"""
    logger.info("\n" + "="*80)
    logger.info("Testing Semantic Pipeline")
    logger.info("="*80)
    
    try:
        from src.semantic.semantic_pipeline import SemanticPipeline
        
        # Create pipeline
        pipeline = SemanticPipeline({'enabled': True})
        logger.info("✓ Semantic pipeline created")
        
        # Test analysis with the correct method name
        test_questions = [
            "How many cars were made in 1980?",
            "What is the average price by manufacturer?",
            "Show me the top 10 fastest cars"
        ]
        
        for question in test_questions:
            # Use the analyze() method
            analysis = pipeline.analyze(question)
            logger.info(f"✓ Analyzed: {question}")
            logger.info(f"  Complexity: {analysis.get('complexity', 'unknown')}")
            
            # Verify the response has expected fields
            assert 'complexity' in analysis, "Missing complexity field"
            assert 'original_question' in analysis, "Missing original_question field"
        
        logger.info("✓ All semantic pipeline tests passed")
        return True, None
        
    except Exception as e:
        logger.error(f"✗ Semantic pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)


def test_trajectory_creation():
    """Test Trajectory dataclass creation"""
    logger.info("\n" + "="*80)
    logger.info("Testing Trajectory Dataclass")
    logger.info("="*80)
    
    try:
        from src.reasoning import ExperienceCollector, Trajectory
        
        # Test creating trajectory with required args only
        trajectory = Trajectory(
            trajectory_id="test_001",
            question="How many cars?",
            database="car_1",
            generated_sql="SELECT COUNT(*) FROM cars"
        )
        logger.info("✓ Trajectory created with required args")
        
        # Test creating trajectory with all args
        trajectory_full = Trajectory(
            trajectory_id="test_002",
            question="What is the average price?",
            database="car_1",
            generated_sql="SELECT AVG(price) FROM cars",
            schema={'tables': ['cars']},
            gold_sql="SELECT AVG(price) FROM cars",
            strategies_used=["strategy_1"],
            reasoning_steps=["step 1", "step 2"],
            generation_time=1.5
        )
        logger.info("✓ Trajectory created with all args")
        
        # Test ExperienceCollector
        collector = ExperienceCollector()
        collector.add_trajectory(trajectory)
        collector.add_trajectory(trajectory_full)
        logger.info(f"✓ ExperienceCollector has {len(collector.get_all_trajectories())} trajectories")
        
        return True, None
        
    except Exception as e:
        logger.error(f"✗ Trajectory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)


def test_reasoning_bank_pipeline():
    """Test ReasoningBank pipeline functionality"""
    logger.info("\n" + "="*80)
    logger.info("Testing ReasoningBank Pipeline")
    logger.info("="*80)
    
    try:
        from src.reasoning import ReasoningBankPipeline, create_reasoning_pipeline
        
        # Create temporary directory for test
        test_dir = Path("./test_memory")
        test_dir.mkdir(exist_ok=True)
        
        # Create pipeline
        pipeline = create_reasoning_pipeline(
            db_path=str(test_dir / "test_reasoning.db"),
            chromadb_path=str(test_dir / "test_chromadb"),
            config={
                'enable_retrieval': True,
                'enable_distillation': True,
                'enable_consolidation': True,
                'consolidation_frequency': 5
            }
        )
        logger.info("✓ ReasoningBank pipeline created")
        
        # Test SQL generator function
        def dummy_sql_generator(question):
            if "how many" in question.lower():
                return "SELECT COUNT(*) FROM cars"
            elif "average" in question.lower():
                return "SELECT AVG(price) FROM cars"
            else:
                return "SELECT * FROM cars"
        
        # Test enhancement
        test_cases = [
            {
                'question': 'How many cars were made in 1980?',
                'db_id': 'car_1',
                'schema': {'tables': ['cars'], 'columns': {'cars': ['id', 'year']}},
                'gold_sql': 'SELECT COUNT(*) FROM cars WHERE year = 1980'
            },
            {
                'question': 'What is the average price?',
                'db_id': 'car_1',
                'schema': {'tables': ['cars'], 'columns': {'cars': ['id', 'price']}},
                'gold_sql': 'SELECT AVG(price) FROM cars'
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            result = pipeline.enhance_sql_generation(
                question=test_case['question'],
                db_id=test_case['db_id'],
                schema=test_case['schema'],
                gold_sql=test_case['gold_sql'],
                sql_generator=dummy_sql_generator
            )
            
            logger.info(f"✓ Test case {i+1} processed")
            logger.info(f"  SQL: {result['sql']}")
            logger.info(f"  Trajectory ID: {result['trajectory_id']}")
            logger.info(f"  Strategies used: {len(result['strategies_used'])}")
            
            # Process evaluation result
            pipeline.process_evaluation_result(
                trajectory_id=result['trajectory_id'],
                exact_match=0.8,
                execution_match=True
            )
            logger.info(f"✓ Evaluation result processed")
        
        # Test consolidation
        pipeline.consolidate_memory()
        logger.info("✓ Memory consolidation completed")
        
        # Get statistics
        stats = pipeline.get_statistics()
        logger.info("\nPipeline Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        # Cleanup
        import shutil
        try:
            shutil.rmtree(test_dir)
            logger.info("✓ Test cleanup completed")
        except:
            logger.warning("Could not clean up test directory")
        
        return True, None
        
    except Exception as e:
        logger.error(f"✗ ReasoningBank pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)


def test_end_to_end_integration():
    """Test complete end-to-end integration"""
    logger.info("\n" + "="*80)
    logger.info("Testing End-to-End Integration")
    logger.info("="*80)
    
    try:
        from src.semantic.semantic_pipeline import SemanticPipeline
        from src.reasoning import ReasoningBankPipeline
        
        # Create temporary directory
        test_dir = Path("./test_e2e")
        test_dir.mkdir(exist_ok=True)
        
        # Initialize pipelines
        semantic_pipeline = SemanticPipeline({'enabled': True})
        reasoning_pipeline = ReasoningBankPipeline(
            db_path=str(test_dir / "reasoning.db"),
            chromadb_path=str(test_dir / "chromadb")
        )
        
        logger.info("✓ Pipelines initialized")
        
        # Dummy SQL generator
        def sql_generator(question):
            return "SELECT COUNT(*) FROM students"
        
        # Test case
        question = "How many students are enrolled?"
        db_id = "university"
        schema = {'tables': ['students'], 'columns': {'students': ['id', 'name']}}
        
        # Step 1: Semantic analysis
        semantic_analysis = semantic_pipeline.analyze(question)
        logger.info("✓ Semantic analysis completed")
        logger.info(f"  Complexity: {semantic_analysis.get('complexity')}")
        
        # Step 2: Generate with ReasoningBank
        result = reasoning_pipeline.enhance_sql_generation(
            question=question,
            db_id=db_id,
            schema=schema,
            semantic_analysis=semantic_analysis,
            sql_generator=sql_generator
        )
        logger.info("✓ SQL generation completed")
        logger.info(f"  Generated SQL: {result['sql']}")
        
        # Step 3: Process evaluation
        reasoning_pipeline.process_evaluation_result(
            trajectory_id=result['trajectory_id'],
            exact_match=1.0,
            execution_match=True
        )
        logger.info("✓ Evaluation processing completed")
        
        # Step 4: Consolidate
        reasoning_pipeline.consolidate_memory()
        logger.info("✓ Memory consolidation completed")
        
        # Get statistics
        stats = reasoning_pipeline.get_statistics()
        logger.info("\nEnd-to-End Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        # Cleanup
        import shutil
        try:
            shutil.rmtree(test_dir)
            logger.info("✓ End-to-end test cleanup completed")
        except:
            logger.warning("Could not clean up test directory")
        
        return True, None
        
    except Exception as e:
        logger.error(f"✗ End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)


def main():
    """Run all integration tests"""
    logger.info("="*80)
    logger.info("REASONINGBANK INTEGRATION TEST SUITE")
    logger.info("="*80)
    
    start_time = time.time()
    
    results = []
    
    # Test 1: Imports
    import_tests = test_imports()
    results.extend(import_tests)
    
    # Test 2: Semantic pipeline
    semantic_result = test_semantic_pipeline()
    results.append(("Semantic Pipeline", *semantic_result))
    
    # Test 3: Trajectory creation
    trajectory_result = test_trajectory_creation()
    results.append(("Trajectory Dataclass", *trajectory_result))
    
    # Test 4: ReasoningBank pipeline
    reasoning_result = test_reasoning_bank_pipeline()
    results.append(("ReasoningBank Pipeline", *reasoning_result))
    
    # Test 5: End-to-end integration
    e2e_result = test_end_to_end_integration()
    results.append(("End-to-End Integration", *e2e_result))
    
    # Summary
    elapsed = time.time() - start_time
    
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for name, success, error in results:
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{status}: {name}")
        if error:
            logger.info(f"       Error: {error}")
    
    logger.info("="*80)
    logger.info(f"Results: {passed}/{total} tests passed")
    logger.info(f"Time: {elapsed:.2f}s")
    logger.info("="*80)
    
    if passed == total:
        logger.info("\n🎉 All tests passed! ReasoningBank integration is working correctly.")
    else:
        logger.warning(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")
    
    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())