# Extra Models Smoke Tests

These scripts allow for quick verification of the additional models (GLM-4.7-Flash and DeepSeek-R1-Distill-Qwen-7B) running via LM Studio.

## Setup
Ensure LM Studio is running on `http://localhost:1234`.

## GLM-4.7-Flash Smoke Test

**File**: `smoke_test_glm_4_7_flash.py`

```python
import requests
import json
import sys

MODEL_ID = "zai-org/glm-4.7-flash:2"
URL = "http://localhost:1234/v1/chat/completions"

def run_smoke_test():
    print(f"Running smoke test for: {MODEL_ID}")
    
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "user", "content": "Hello. Are you working?"}
        ],
        "temperature": 0.7,
        "max_tokens": 50,
        "stream": False
    }
    
    try:
        response = requests.post(URL, json=payload, timeout=30)
        
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            print("✅ SUCCESS")
            print(f"Response: {content}")
        else:
            print(f"❌ FAILED (Status {response.status_code})")
            print(response.text)
            
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    run_smoke_test()
```

## DeepSeek-R1-Distill-Qwen-7B Smoke Test

**File**: `smoke_test_deepseek_r1_distill_qwen_7b.py`

```python
import requests
import json
import sys

MODEL_ID = "deepseek-r1-distill-qwen-7b"
URL = "http://localhost:1234/v1/chat/completions"

def run_smoke_test():
    print(f"Running smoke test for: {MODEL_ID}")
    
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "user", "content": "Hello. Are you working?"}
        ],
        "temperature": 0.7,
        "max_tokens": 50,
        "stream": False
    }
    
    try:
        response = requests.post(URL, json=payload, timeout=30)
        
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            print("✅ SUCCESS")
            print(f"Response: {content}")
        else:
            print(f"❌ FAILED (Status {response.status_code})")
            print(response.text)
            
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    run_smoke_test()
```
