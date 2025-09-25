#!/usr/bin/env python3
import os
import sys
from pathlib import Path

def main():
    model_id = os.getenv("MODEL_ID")
    model_dir = Path(os.getenv("MODEL_DIR", "gpt-oss/gpt-oss-20b"))
    token = os.getenv("HF_TOKEN")

    if not model_id:
        print("ERROR: MODEL_ID env var is required (e.g., meta-llama/Llama-2-7b-hf)", file=sys.stderr)
        sys.exit(2)

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        print("Installing huggingface_hub ...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "huggingface_hub"])
        from huggingface_hub import snapshot_download

    model_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {"repo_id": model_id, "local_dir": str(model_dir), "local_dir_use_symlinks": False}
    if token:
        kwargs["token"] = token

    print(f"Downloading {model_id} -> {model_dir}")
    snapshot_download(**kwargs)
    print("Download complete.")

if __name__ == "__main__":
    main()
