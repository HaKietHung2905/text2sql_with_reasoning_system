import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Add project root to sys.path
current_dir = Path(__file__).parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import vertexai
from google.cloud import aiplatform

def list_models():
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = "us-central1"
    
    print(f"Checking models for project: {project_id} in {location}")
    
    try:
        aiplatform.init(project=project_id, location=location)
        
        # This lists all models, including publisher models if possible
        # Note: listing foundational models often requires specific API calls
        print("\n--- Model Garden Models ---")
        # In newer SDKs, we might need to use the Model Garden client
        from google.cloud.aiplatform_v1 import ModelGardenServiceClient
        client = ModelGardenServiceClient(client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"})
        
        # Try to search for Gemini
        request = {
            "parent": f"locations/{location}",
            "filter": 'publisher="google"',
        }
        
        # Note: This might not work in all environments, but it's worth a try
        # Actually, let's just try to list instances of GenerativeModel if we can
        print("Attempting to list models via ModelGardenServiceClient...")
        # (This is just a placeholder, listing publisher models is actually hard via SDK)
        
        print("\nTry to call generate_content on specific models:")
        models_to_test = ["gemini-1.5-flash", "gemini-1.5-pro"]
        for m in models_to_test:
            try:
                from vertexai.generative_models import GenerativeModel
                model = GenerativeModel(m)
                print(f"  Testing '{m}'...")
                response = model.generate_content("Say hi")
                print(f"    [SUCCESS] Response: {response.text}")
            except Exception as e:
                print(f"    [FAIL] Model '{m}' failed during generation: {e}")
                
        # diagnostic: check if we can even list endpoints
        print("\nChecking if we can list endpoints (checks general API state):")
        try:
            endpoints = aiplatform.Endpoint.list()
            print(f"  [OK] Successfully listed {len(endpoints)} endpoints.")
        except Exception as e:
            print(f"  [FAIL] Endpoint list failed: {e}")

    except Exception as e:
        print(f"Diagnostic failed: {e}")

if __name__ == "__main__":
    list_models()
