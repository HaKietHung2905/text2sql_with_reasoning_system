import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Load environment variables
load_dotenv()

from src.models.google_genai import GoogleGenAI

def test_vertex_ai_connection():
    print("Testing Vertex AI Connection...")
    
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    if not project_id:
        print("❌ GOOGLE_CLOUD_PROJECT not found in environment")
        print("   Set it in .env file or export GOOGLE_CLOUD_PROJECT=your-project-id")
        return False
    
    if not credentials_path:
        print("⚠️  GOOGLE_APPLICATION_CREDENTIALS not set")
        print("   Will attempt to use Application Default Credentials")
    elif not os.path.exists(credentials_path):
        print(f"❌ Credentials file not found: {credentials_path}")
        return False
    else:
        print(f"✓ Found credentials file: {credentials_path}")
    
    print(f"✓ Found Project ID: {project_id}")
    
    try:
        print("\nInitializing Vertex AI...")
        model = GoogleGenAI()
        
        print("Generating test content...")
        response = model.generate("Say 'Hello from Vertex AI' if you can hear me.")
        print(f"✓ Generation Successful!")
        print(f"Response: {response.strip()}")
        return True
        
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure Vertex AI API is enabled in your GCP project")
        print("2. Verify service account has 'Vertex AI User' role")
        print("3. Check that GOOGLE_APPLICATION_CREDENTIALS points to valid JSON key")
        return False

if __name__ == "__main__":
    success = test_vertex_ai_connection()
    sys.exit(0 if success else 1)
