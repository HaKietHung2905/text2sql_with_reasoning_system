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
    
    # Create test directory
    test_dir = Path("./test_trajectory")
    test_dir.mkdir(exist_ok=True)
    
    try:
        # Initialize pipeline
        pipeline = ReasoningBankPipeline(
            db_path=str(test_dir / "test.db"),
            chromadb_path=str(test_dir / "chroma")
        )
        print("\n✓ Pipeline initialized")
        
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
            print(f"   Judgment success: {judgment.is_success()}")
            print(f"   Judgment score: {judgment.overall_score}")
        else:
            print(f"   ⚠️  No judgment found")
        
        # Get statistics
        print(f"\n📈 Pipeline Statistics:")
        stats = pipeline.get_statistics()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Cleanup
        shutil.rmtree(test_dir)
        print("\n✓ Test cleanup complete")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on error
        if test_dir.exists():
            shutil.rmtree(test_dir)
        
        return False

if __name__ == "__main__":
    success = test_trajectory_storage()
    print("\n" + "="*80)
    if success:
        print("✅ ALL TESTS PASSED - Trajectory storage is working!")
    else:
        print("❌ TESTS FAILED - Trajectory storage has issues")
    print("="*80)
    
    sys.exit(0 if success else 1)