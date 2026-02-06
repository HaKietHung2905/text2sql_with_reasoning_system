"""
Test if trajectory storage is working
"""
import sys
sys.path.append('.')

from src.reasoning import ReasoningBankPipeline
from pathlib import Path
import shutil

def test_trajectory_storage():
    print("="*80)
    print("TESTING TRAJECTORY STORAGE FLOW")
    print("="*80)
    
    # Use the ACTUAL seeded database, not a test one
    print("\n✓ Using production database with seeded strategies")
    
    try:
        # Initialize pipeline with ACTUAL database
        pipeline = ReasoningBankPipeline(
            db_path="./memory/reasoning_bank.db",  
            chromadb_path="./memory/chromadb"    
        )
        print("\n✓ Pipeline initialized")
        
        # Check what's in memory FIRST
        stats = pipeline.memory_store.get_statistics()
        print(f"\n📊 Memory contains {stats['total_strategies']} strategies")
        
        if stats['total_strategies'] == 0:
            print("\n⚠️  WARNING: No strategies found! Run seed_initial_strategies.py first")
            return False
        
        # Create a simple SQL generator
        def sql_gen(q):
            return "SELECT COUNT(*) FROM students"
        
        # Test SQL generation
        print("\n📝 Testing SQL generation...")
        result = pipeline.enhance_sql_generation(
            question="How many students are there?",
            db_id="university",
            schema={'tables': ['students'], 'columns': {'students': ['id', 'name']}},
            gold_sql="SELECT COUNT(*) FROM students",
            sql_generator=sql_gen
        )
        
        print(f"\n✅ Generation Result:")
        print(f"   SQL: {result['sql']}")
        print(f"   Trajectory ID: {result['trajectory_id']}")
        print(f"   Strategies used: {result['strategies_used']}")
        
        if not result['strategies_used']:
            print(f"\n⚠️  No strategies were used - checking why...")
            
            # Debug retrieval
            from src.reasoning.memory_retrieval import RetrievalContext
            context = RetrievalContext(
                query="How many students are there?",
                database="university",
                schema={'tables': ['students'], 'columns': {'students': ['id', 'name']}}
            )
            
            patterns = pipeline.memory_retrieval._identify_candidate_patterns(context)
            print(f"   Detected patterns: {patterns}")
            
            for pattern in patterns:
                strategies = pipeline.memory_store.get_strategies_by_pattern(pattern)
                print(f"   Pattern '{pattern}': {len(strategies)} strategies")
        
        # Check if trajectory is stored
        trajectory_id = result['trajectory_id']
        trajectory = pipeline.experience_collector.get_trajectory(trajectory_id)
        
        if trajectory:
            print(f"\n✅ Trajectory Retrieved:")
            print(f"   ID: {trajectory.trajectory_id}")
            print(f"   Question: {trajectory.question}")
            print(f"   Database: {trajectory.database}")
            print(f"   Generated SQL: {trajectory.generated_sql}")
            print(f"   Gold SQL: {trajectory.gold_sql}")
            print(f"   Strategies used: {trajectory.strategies_used}")
        else:
            print(f"\n❌ Trajectory NOT found!")
            return False
        
        # Test evaluation feedback
        print(f"\n📊 Testing evaluation feedback...")
        pipeline.process_evaluation_result(
            trajectory_id=trajectory_id,
            exact_match=0.8,
            execution_match=True
        )
        
        # Check if judgment was added
        trajectory = pipeline.experience_collector.get_trajectory(trajectory_id)
        print(f"\n✅ After Evaluation:")
        print(f"   Exact match: {trajectory.exact_match}")
        print(f"   Execution match: {trajectory.execution_match}")
        
        judgment = pipeline.experience_collector.get_judgment(trajectory_id)
        if judgment:
            print(f"\n✅ Judgment Details:")
            print(f"   Trajectory ID: {judgment.trajectory_id}")
            print(f"   Success type: {judgment.success_type.value}")
            print(f"   Is success: {judgment.is_success()}")
            print(f"   Is complete success: {judgment.is_complete_success()}")
            print(f"   Confidence: {judgment.confidence}")
            print(f"   Number of insights: {len(judgment.insights)}")
        else:
            print(f"   ⚠️  No judgment found")
        
        # Get statistics
        print(f"\n📈 Pipeline Statistics:")
        stats = pipeline.get_statistics()
        for key, value in stats.items():
            if key != 'memory_store_stats':  # Skip nested dict for cleaner output
                print(f"   {key}: {value}")
        
        # Check experience collector stats
        exp_stats = pipeline.experience_collector.get_statistics()
        print(f"\n📊 Experience Collector Statistics:")
        for key, value in exp_stats.items():
            print(f"   {key}: {value}")
        
        print("\n" + "="*80)
        if result['strategies_used']:
            print("✅ ALL TESTS PASSED - STRATEGIES WORKING!")
        else:
            print("⚠️  TESTS PASSED BUT NO STRATEGIES RETRIEVED")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_trajectory_storage()
    
    if not success:
        sys.exit(1)