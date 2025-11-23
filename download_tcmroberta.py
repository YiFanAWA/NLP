"""Download a default Chinese pretrained model and save it to ./TCMroberta.

Usage (PowerShell):
  cd TCMERE/TCMERE-main
  python download_tcmroberta.py

Notes:
- This will download model files (hundreds of MB). Ensure you have network access and disk space.
- If you don't have `transformers` installed, install it first: `pip install transformers`.
"""
import sys
import os

MODEL_ID = "hfl/chinese-roberta-wwm-ext"
TARGET_DIR = os.path.join(os.path.dirname(__file__), "TCMroberta")


def main():
    try:
        from transformers import AutoTokenizer, AutoModel
    except Exception as e:
        print("Missing transformers. Install with: pip install transformers", file=sys.stderr)
        raise

    print(f"Downloading model {MODEL_ID} to {TARGET_DIR}...")
    os.makedirs(TARGET_DIR, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)
    tok.save_pretrained(TARGET_DIR)
    model.save_pretrained(TARGET_DIR)
    print("Download complete. Saved to:")
    for f in sorted(os.listdir(TARGET_DIR)):
        print(' -', f)


if __name__ == '__main__':
    main()
