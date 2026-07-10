#!/usr/bin/env python3
import os
import urllib.request
import argparse
from pathlib import Path

def download_model(force: bool = False) -> None:
    url = "https://huggingface.co/SupraLabs/Supra-Router-51M-gguf/resolve/main/Supra-Router-51M-Q4_K_M.gguf"
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    path = models_dir / "Supra-Router-51M-Q4_K_M.gguf"
    if path.exists() and not force:
        print(f"[*] ML Router already exists at {path}")
        return
        
    print(f"[*] Downloading Supra-Router-51M to {path}...")
    urllib.request.urlretrieve(url, path)
    print("[*] Download complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force redownload")
    args = parser.parse_args()
    download_model(args.force)
