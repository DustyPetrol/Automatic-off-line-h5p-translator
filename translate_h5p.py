import os
import json
import shutil
import zipfile
from bs4 import BeautifulSoup
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch

# --------------------------
# Local AI Setup
# --------------------------
MODEL_NAME = "facebook/m2m100_418M"
SOURCE_LANG = "en"
TARGET_LANG = "de"

print("[INFO] Loading local AI model... (first time may take ~30s)")

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME)
model = M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
tokenizer.src_lang = SOURCE_LANG

def translate_local_ai(text):
    encoded = tokenizer(text, return_tensors="pt").to(device)
    generated = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.lang_code_to_id[TARGET_LANG]
    )
    return tokenizer.decode(generated[0], skip_special_tokens=True)

# --------------------------
# HTML-Aware Translation
# --------------------------
def translate_html_preserve_tags(html, translator_func):
    soup = BeautifulSoup(html, "html.parser")
    for node in soup.find_all(string=True):
        if node.strip():
            try:
                node.replace_with(translator_func(node))
            except Exception as e:
                print(f"[WARN] Failed to translate part: {node[:30]} - {e}")
    return str(soup)

# --------------------------
# Deep Translation Walker
# --------------------------
def translate_json_fields(data, translator_func):
    keys_to_translate = {"text", "alt", "title", "contentName", "question", "header",
                        "body", "checkAnswerButton", "submitAnswerButton", "showSolutionButton",
                        "a11yCheck", "a11yShowSolution", "a11yRetry", "feedbackOnWrong"}

    if isinstance(data, dict):
        for key, value in data.items():
            if key in keys_to_translate and isinstance(value, str):
                try:
                    # Detect HTML and preserve tags if needed
                    if "<" in value and "</" in value:
                        translated = translate_html_preserve_tags(value, translator_func)
                    else:
                        translated = translator_func(value)
                    data[key] = translated
                except Exception as e:
                    print(f"[WARN] Couldn't translate {key}: {e}")
            else:
                translate_json_fields(value, translator_func)

    elif isinstance(data, list):
        for item in data:
            translate_json_fields(item, translator_func)


# --------------------------
# H5P Translator
# --------------------------
def translate_h5p(input_h5p, output_h5p):
    temp_dir = "temp_h5p"
    content_json_path = os.path.join(temp_dir, "content", "content.json")

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    with zipfile.ZipFile(input_h5p, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    print("[OK] Extracted H5P")

    with open(content_json_path, 'r', encoding='utf-8') as f:
        content = json.load(f)

    print("[INFO] Translating content using local AI...")
    translate_json_fields(content, translate_local_ai)

    with open(content_json_path, 'w', encoding='utf-8') as f:
        json.dump(content, f, ensure_ascii=False, indent=2)

    with zipfile.ZipFile(output_h5p, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                filepath = os.path.join(root, file)
                arcname = os.path.relpath(filepath, temp_dir)
                zipf.write(filepath, arcname)

    print(f"[✅] Translated and saved: {output_h5p}")
    shutil.rmtree(temp_dir)

# --------------------------
# Run via CLI
# --------------------------
if __name__ == "__main__":
    import sys
    input_file = sys.argv[1] if len(sys.argv) > 1 else "freshman-laboratory-safety.h5p"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "freshman-laboratory-safety-GER.h5p"
    translate_h5p(input_file, output_file)
