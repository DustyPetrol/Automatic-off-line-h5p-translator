import os
import json
import shutil
import zipfile
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from bs4 import BeautifulSoup
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch
import threading

# === Local AI Translator Setup ===
MODEL_NAME = "facebook/m2m100_418M"
SOURCE_LANG = "en"
TARGET_LANG = "de"
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

def translate_html_preserve_tags(html, translator_func, log_callback):
    from bs4 import BeautifulSoup

    def safe_translate(block_html):
        """Try translating a block, with retries and fallback."""
        attempts = 2
        for i in range(attempts):
            try:
                translated = translator_func(block_html)
                return translated if translated.strip() else block_html
            except Exception as e:
                log_callback(f"[Retry {i+1}/{attempts}] Error: {str(e)[:80]}")
        return block_html  # fallback

    try:
        soup = BeautifulSoup(html, "html.parser")
        blocks = soup.find_all(['ol', 'ul', 'p', 'li', 'h1', 'h2', 'h3', 'h4'])
        if not blocks:
            blocks = [soup]

        translated_fragments = []

        for block in blocks:
            block_html = str(block).strip()
            if not block_html:
                continue

            translated = safe_translate(block_html)

            original_len = len(BeautifulSoup(block_html, "html.parser").text.strip())
            translated_len = len(BeautifulSoup(translated, "html.parser").text.strip())

            if translated_len < 0.5 * original_len:
                log_callback(f"[WARN] Truncated block (keeping original) → {translated_len} vs {original_len}")
                translated_fragments.append(block_html)
            else:
                translated_fragments.append(translated)
                log_callback(f"[Block] {block.text.strip()} → {BeautifulSoup(translated, 'html.parser').text.strip()[:80]}")

        return "\n".join(translated_fragments)

    except Exception as e:
        log_callback(f"[FATAL] Entire HTML block failed: {html[:60]}... ({e})")
        return html




def translate_json_fields(data, translator_func, log_callback, translated_flags=None, current_path="root"):
    if translated_flags is None:
        translated_flags = set()

    translatable_keys = {"text", "question", "title", "alt", "label", "contentName"}

    if isinstance(data, dict):
        for key, value in data.items():
            path = f"{current_path}/{key}"
            if path in translated_flags:
                continue

            if key == "answers" and isinstance(value, list):
                for idx, answer in enumerate(value):
                    answer_path = f"{path}[{idx}]"
                    if isinstance(answer, dict) and "text" in answer:
                        orig = answer["text"]
                        sub_path = f"{answer_path}/text"
                        if sub_path in translated_flags:
                            continue
                        if isinstance(orig, str) and orig.strip():
                            try:
                                translated = translator_func(orig)
                                answer["text"] = translated
                                translated_flags.add(sub_path)
                                log_callback(f"[Answer] {orig.strip()} → {translated.strip()}")
                            except Exception as e:
                                log_callback(f"[Error] Translating answer[{idx}] failed: {orig[:40]}... ({e})")
                    elif isinstance(answer, (dict, list)):
                        translate_json_fields(answer, translator_func, log_callback, translated_flags, answer_path)
            elif isinstance(value, dict) or isinstance(value, list):
                translate_json_fields(value, translator_func, log_callback, translated_flags, path)
            elif key in translatable_keys and isinstance(value, str) and value.strip():
                try:
                    if "<" in value and ">" in value:
                        translated = translate_html_preserve_tags(value, translator_func, log_callback)
                    else:
                        translated = translator_func(value)
                        log_callback(f"{value.strip()} → {translated.strip()}")
                    data[key] = translated
                    translated_flags.add(path)
                except Exception as e:
                    log_callback(f"[WARN] Couldn't translate {key} at {path}: {e}")

    elif isinstance(data, list):
        for idx, item in enumerate(data):
            path = f"{current_path}[{idx}]"
            if isinstance(item, (dict, list)):
                translate_json_fields(item, translator_func, log_callback, translated_flags, path)
            elif isinstance(item, str) and path not in translated_flags:
                try:
                    translated = translator_func(item)
                    log_callback(f"{item.strip()} → {translated.strip()}")
                    data[idx] = translated
                    translated_flags.add(path)
                except Exception as e:
                    log_callback(f"[WARN] Couldn't translate list item at {path}: {e}")

def translate_h5p(input_h5p, output_h5p, log_callback, export_raw=False):
    temp_dir = "temp_h5p"
    content_json_path = os.path.join(temp_dir, "content", "content.json")

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    with zipfile.ZipFile(input_h5p, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    log_callback("[OK] Extracted H5P")

    with open(content_json_path, 'r', encoding='utf-8') as f:
        content = json.load(f)

    log_callback("[INFO] Starting translation...")
    translate_json_fields(content, translate_local_ai, log_callback)

    with open(content_json_path, 'w', encoding='utf-8') as f:
        json.dump(content, f, ensure_ascii=False, indent=2)

    if export_raw:
        shutil.make_archive("file-we-just-translated", 'zip', temp_dir)

    with zipfile.ZipFile(output_h5p, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(temp_dir):
            for file in files:
                path = os.path.join(root, file)
                arcname = os.path.relpath(path, temp_dir)
                zipf.write(path, arcname)

    log_callback(f"[✅] Translated and saved: {output_h5p}")
    shutil.rmtree(temp_dir)

# === GUI ===
class TranslatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("H5P Translator (Local AI)")
        self.root.geometry("800x650")

        self.file_path = tk.StringVar()
        self.export_raw = tk.BooleanVar()
        self.log_file_path = "translation-log.txt"

        tk.Button(root, text="Select .h5p File", command=self.select_file).pack(pady=5)
        tk.Entry(root, textvariable=self.file_path, width=100).pack()

        tk.Checkbutton(root, text="Export translated folder as ZIP (for debugging)", variable=self.export_raw).pack(pady=5)

        tk.Button(root, text="Translate Now", command=self.start_translation).pack(pady=5)

        self.log = scrolledtext.ScrolledText(root, height=25, width=100)
        self.log.pack(pady=10)

        self.status = tk.Label(root, text="Idle", fg="blue")
        self.status.pack()

    def log_msg(self, message):
        self.log.insert(tk.END, message + "\n")
        self.log.see(tk.END)
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")
        self.root.update()

    def select_file(self):
        path = filedialog.askopenfilename(filetypes=[("H5P files", "*.h5p")])
        if path:
            self.file_path.set(path)

    def start_translation(self):
        file = self.file_path.get()
        if not file or not file.endswith(".h5p"):
            messagebox.showerror("Error", "Please select a valid .h5p file.")
            return

        output_file = file.replace(".h5p", "_translated.h5p")
        if os.path.exists(output_file):
            overwrite = messagebox.askyesno("Overwrite?", f"{output_file} already exists. Overwrite?")
            if not overwrite:
                self.status.config(text="Cancelled", fg="orange")
                return

        self.status.config(text="Translating...", fg="black")
        self.log.delete("1.0", tk.END)
        with open(self.log_file_path, "w", encoding="utf-8") as f:
            f.write(f"Translating file: {file}\n")

        threading.Thread(target=self.run_translation, args=(file, output_file, self.export_raw.get())).start()

    def run_translation(self, input_file, output_file, export_raw):
        try:
            translate_h5p(input_file, output_file, self.log_msg, export_raw)
            self.status.config(text="✅ Translation complete", fg="green")
        except Exception as e:
            self.log_msg(f"[ERROR] {e}")
            self.status.config(text="❌ Error occurred", fg="red")

# === Run GUI App ===
if __name__ == "__main__":
    root = tk.Tk()
    app = TranslatorGUI(root)
    root.mainloop()
