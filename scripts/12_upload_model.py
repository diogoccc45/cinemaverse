"""
SCRIPT 12 - Upload fine-tuned GPT-2 to Hugging Face Hub
=========================================================
Run manually or via GitHub Actions (uses HF_TOKEN secret).

How to run locally:
    HF_TOKEN=your_token python scripts/12_upload_model.py
"""

import os
from huggingface_hub import HfApi, login

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.join(SCRIPT_DIR, '..')
MODEL_DIR  = os.path.join(ROOT_DIR, 'models', 'letterboxd_gpt2')
HF_REPO    = "diogocc/letterboxd-gpt2"

# Token comes from environment variable — never hardcoded
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set. "
                     "Set it via GitHub Secrets or export HF_TOKEN=... locally.")

login(token=HF_TOKEN)

api = HfApi()

print(f"Creating repo: {HF_REPO}")
api.create_repo(repo_id=HF_REPO, private=False, exist_ok=True)

print(f"Uploading model from: {MODEL_DIR}")
api.upload_folder(
    folder_path=MODEL_DIR,
    repo_id=HF_REPO,
    repo_type="model",
)

print("=" * 50)
print(f"  Model at: https://huggingface.co/{HF_REPO}")
print("=" * 50)
