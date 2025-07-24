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
def sanitize_html(html):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    return str(soup)

def translate_local_ai(text):
    encoded = tokenizer(text, return_tensors="pt").to(device)
    generated = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.lang_code_to_id[TARGET_LANG]
    )
    return tokenizer.decode(generated[0], skip_special_tokens=True)

def translate_html_preserve_tags(html, translator_func, log_callback, max_tokens=400):
    from bs4 import BeautifulSoup

    def estimate_tokens(text):
        # Roughly 4 chars per token
        return max(1, len(text) // 4)
    
    html = sanitize_html(html)
    if estimate_tokens(html) <= max_tokens:
        try:
            translated = translator_func(html)
            log_callback(f"[HTML] {BeautifulSoup(html, 'html.parser').get_text().strip()} → {BeautifulSoup(translated, 'html.parser').get_text().strip()[:80]}")
            return translated
        except Exception as e:
            log_callback(f"[FATAL] HTML translation failed: {html[:60]}... ({e})")
            return html

    soup = BeautifulSoup(html, "html.parser")
    translated_html = ""

    # Try to find a single parent tag that contains all content
    parent = None
    for tag in soup.contents:
        if hasattr(tag, "name"):
            parent = tag
            break
    
    if parent and parent.name in ["ol", "ul"]:
        # Translate each <li> as an HTML string, then reassemble
        for li in parent.find_all("li", recursive=False):
            li_html = str(li)
            if estimate_tokens(li_html) > max_tokens:
                translated_li = translate_html_preserve_tags(li_html, translator_func, log_callback, max_tokens)
            else:
                try:
                    translated_li = translator_func(li_html)
                except Exception as e:
                    log_callback(f"[LI ERROR] {e}")
                    translated_li = li_html
            li.clear()
            li.append(BeautifulSoup(translated_li, "html.parser"))
        translated_html = str(soup)
    else:
        # Otherwise, translate each top-level <p> or fallback to lines
        blocks = soup.find_all("p", recursive=False)
        if not blocks:
            # fallback: lines
            blocks = [BeautifulSoup(f"<p>{line}</p>", "html.parser") for line in html.split('\n') if line.strip()]
        for block in blocks:
            block_html = str(block)
            if estimate_tokens(block_html) > max_tokens:
                translated_block = translate_html_preserve_tags(block_html, translator_func, log_callback, max_tokens)
            else:
                try:
                    translated_block = translator_func(block_html)
                except Exception as e:
                    log_callback(f"[P ERROR] {e}")
                    translated_block = block_html
            block.clear()
            block.append(BeautifulSoup(translated_block, "html.parser"))
        translated_html = str(soup)

    return translated_html

    # Translate direct text in <li> and <p> only, not tags or children tags
    def translate_tag_text(tag):
        for child in tag.children:
            if isinstance(child, str):
                raw = child.strip()
                if raw:
                    try:
                        translated = translator_func(raw)
                        tag.string.replace_with(translated)
                    except Exception as e:
                        log_callback(f"[TEXT ERROR] {e}")
                        continue
            elif hasattr(child, 'text') and child.text.strip():
                try:
                    # Recursively translate the text in child tags if not too big
                    if estimate_tokens(child.text) <= max_tokens:
                        translated = translator_func(child.text)
                        child.string.replace_with(translated)
                    else:
                        # If too big, recurse
                        translate_tag_text(child)
                except Exception as e:
                    log_callback(f"[TAG ERROR] {e}")
                    continue

    # If soup contains <ol> or <ul>, process each <li>
    for tag in soup.find_all(['ol', 'ul']):
        for li in tag.find_all('li', recursive=False):
            translate_tag_text(li)

    # Also translate top-level <p> tags
    for p in soup.find_all('p', recursive=False):
        translate_tag_text(p)

    return str(soup)

def translate_json_fields(data, translator_func, log_callback, translated_flags=None, current_path="root"):
    if translated_flags is None:
        translated_flags = set()

    translatable_keys = {"text", "question", "title", "alt", "label", "contentName"}

    if isinstance(data, dict):
        for key, value in data.items():
            path = f"{current_path}/{key}"
            if path in translated_flags:
                continue

            # Only handle translation here; do not recurse further for these keys
            if key in translatable_keys and isinstance(value, str) and value.strip():
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
                continue

            # Special case for 'answers' key
            if key == "answers" and isinstance(value, list):
                for idx, answer in enumerate(value):
                    answer_path = f"{path}[{idx}]"
                    if isinstance(answer, dict) and "text" in answer:
                        sub_path = f"{answer_path}/text"
                        if sub_path not in translated_flags:
                            orig = answer["text"]
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
                continue

            # For nested dict/list, recurse
            if isinstance(value, (dict, list)):
                translate_json_fields(value, translator_func, log_callback, translated_flags, path)

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