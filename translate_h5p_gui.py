import os
import json
import shutil
import zipfile
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from bs4 import BeautifulSoup, NavigableString
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch
import threading
import re

# === Local AI Translator Setup ===
MODEL_NAME = "facebook/m2m100_418M"
SOURCE_LANG = "en"
TARGET_LANG = "de"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME)
model = M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
tokenizer.src_lang = SOURCE_LANG

def translate_local_ai(text):
    """Translation with length and quality controls"""
    if not text or not text.strip():
        return text
    
    # Clean up input text
    text = text.strip()
    
    # Limit input length to prevent truncation
    max_input_length = 200  # Conservative limit
    if len(text) > max_input_length:
        # Split by sentences and translate parts
        sentences = re.split(r'([.!?]+)', text)
        translated_parts = []
        current_chunk = ""
        
        for i in range(0, len(sentences), 2):
            sentence = sentences[i] if i < len(sentences) else ""
            punctuation = sentences[i+1] if i+1 < len(sentences) else ""
            full_sentence = sentence + punctuation
            
            if len(current_chunk + full_sentence) > max_input_length:
                if current_chunk:
                    translated_parts.append(translate_single_chunk(current_chunk.strip()))
                    current_chunk = full_sentence
                else:
                    # Single sentence too long, translate as-is
                    translated_parts.append(translate_single_chunk(full_sentence.strip()))
            else:
                current_chunk += full_sentence
        
        if current_chunk:
            translated_parts.append(translate_single_chunk(current_chunk.strip()))
        
        return " ".join(translated_parts)
    else:
        return translate_single_chunk(text)

def translate_single_chunk(text):
    """Translate a single chunk with error handling"""
    try:
        log_text = text[:50] + "..." if len(text) > 50 else text
        print(f"[TRANSLATE-DEBUG] Input: '{log_text}'")
        
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
        generated = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.lang_code_to_id[TARGET_LANG],
            max_length=400,  # Increased max length
            num_beams=3,     # Better quality
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id
        )
        result = tokenizer.decode(generated[0], skip_special_tokens=True)
        
        log_result = result[:50] + "..." if len(result) > 50 else result
        print(f"[TRANSLATE-DEBUG] Output: '{log_result}'")
        
        # Verify result is reasonable
        if len(result.strip()) < 3:
            print(f"[TRANSLATE-DEBUG] Result too short, using original")
            return text  # Fallback to original
        
        # Check if translation actually happened
        if result.strip().lower() == text.strip().lower():
            print(f"[TRANSLATE-DEBUG] No translation occurred, result identical to input")
            
        return result
    except Exception as e:
        print(f"[TRANSLATE-ERROR] Translation error: {e}")
        return text

def extract_and_translate_text_nodes(soup, translator_func, log_callback):
    """Extract text nodes, translate them, and put them back with proper spacing"""
    text_nodes = []
    
    # Find all text nodes, preserving original spacing
    def find_text_nodes(element):
        for child in element.children:
            if isinstance(child, NavigableString):
                original_text = str(child)
                # Keep nodes that have meaningful content (including whitespace-only if needed for spacing)
                if original_text and child.parent.name not in ['script', 'style']:
                    text_nodes.append({
                        'node': child,
                        'text': original_text,
                        'stripped': original_text.strip(),
                        'leading_space': original_text.startswith(' ') or original_text.startswith('\t'),
                        'trailing_space': original_text.endswith(' ') or original_text.endswith('\t')
                    })
            elif hasattr(child, 'children'):
                find_text_nodes(child)
    
    find_text_nodes(soup)
    
    # Translate each meaningful text node
    for text_info in text_nodes:
        stripped_text = text_info['stripped']
        if stripped_text and len(stripped_text) > 1:
            try:
                translated_text = translator_func(stripped_text)
                
                # Preserve original spacing
                final_text = translated_text
                if text_info['leading_space']:
                    final_text = ' ' + final_text
                if text_info['trailing_space']:
                    final_text = final_text + ' '
                
                log_callback(f"[TEXT] '{text_info['text']}' → '{final_text}'")
                text_info['node'].replace_with(final_text)
            except Exception as e:
                log_callback(f"[WARN] Failed to translate: {stripped_text[:30]}... ({e})")

def translate_html_by_element_context(html, translator_func, log_callback):
    """Translate by element context, preserving inline formatting"""
    if not html or not html.strip():
        return html
    
    # Handle plain text
    if "<" not in html or ">" not in html:
        translated = translator_func(html)
        log_callback(f"[PLAIN] {html[:40]}... → {translated[:40]}...")
        return translated
    
    try:
        soup = BeautifulSoup(html, "html.parser")
        log_callback(f"[HTML-DEBUG] Processing HTML: {html[:60]}...")
        
        # Find elements with text content to translate
        translated_any = False
        for element in soup.find_all(['li', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div']):
            # Get full text content
            full_text = element.get_text()
            if full_text.strip() and len(full_text.strip()) > 1:
                try:
                    log_callback(f"[HTML-DEBUG] Translating element text: '{full_text[:50]}...'")
                    # Translate the complete text
                    translated_text = translator_func(full_text.strip())
                    log_callback(f"[HTML-DEBUG] Translation result: '{translated_text[:50]}...'")
                    
                    # Check if translation actually changed
                    if translated_text.strip() != full_text.strip():
                        log_callback(f"[HTML-DEBUG] Translation changed, updating element")
                        # Clear and rebuild with basic structure
                        element.clear()
                        element.string = translated_text
                        translated_any = True
                    else:
                        log_callback(f"[HTML-DEBUG] Translation unchanged, keeping original")
                    
                except Exception as e:
                    log_callback(f"[HTML-ERROR] Element translation failed: {e}")
        
        result = str(soup)
        log_callback(f"[HTML-DEBUG] Final result: {result[:60]}...")
        
        if translated_any:
            return result
        else:
            log_callback(f"[HTML-WARN] No elements were translated, trying simple text extraction")
            # Fallback: just extract and translate the plain text
            plain_text = soup.get_text().strip()
            if plain_text:
                try:
                    translated_plain = translator_func(plain_text)
                    if translated_plain.strip() != plain_text:
                        log_callback(f"[HTML-FALLBACK] Using plain text translation")
                        return f"<div>{translated_plain}</div>"
                except Exception as e:
                    log_callback(f"[HTML-FALLBACK-ERROR] {e}")
            return html
        
    except Exception as e:
        log_callback(f"[HTML-ERROR] Element translation failed: {e}")
        return html

def translate_html_simple_fallback(html, translator_func, log_callback):
    """Ultra-simple fallback: just translate the plain text content"""
    try:
        soup = BeautifulSoup(html, "html.parser")
        plain_text = soup.get_text()
        
        if not plain_text.strip():
            return html
        
        # Translate just the text
        translated_text = translator_func(plain_text)
        log_callback(f"[FALLBACK] {plain_text[:40]}... → {translated_text[:40]}...")
        
        # Try to put it back in similar structure
        if soup.find('ol'):
            # It's a list - split by likely list items
            items = re.split(r'\n\s*(?=\d+\.|\w+\s*\()', translated_text)
            result = "<ol>"
            for item in items:
                if item.strip():
                    result += f"<li><p>{item.strip()}</p></li>"
            result += "</ol>"
            return result
        elif soup.find('p'):
            # Wrap in paragraphs
            paragraphs = translated_text.split('\n\n')
            result = ""
            for para in paragraphs:
                if para.strip():
                    result += f"<p>{para.strip()}</p>"
            return result
        else:
            return f"<p>{translated_text}</p>"
    
    except Exception as e:
        log_callback(f"[ERROR] Fallback translation failed: {e}")
        return html

def translate_html_robust(html, translator_func, log_callback):
    """Robust HTML translation with multiple fallback strategies"""
    if not html or not html.strip():
        return html
    
    # Strategy 1: Element context translation
    try:
        result = translate_html_by_element_context(html, translator_func, log_callback)
        
        # Validate result
        test_soup = BeautifulSoup(result, "html.parser")
        if test_soup.get_text().strip() and len(test_soup.get_text()) > 10:
            return result
        else:
            log_callback("[WARN] Strategy 1 failed, trying text extraction")
    except Exception as e:
        log_callback(f"[WARN] Strategy 1 error: {e}")
    
    # Strategy 2: Text node extraction (original approach)
    try:
        soup = BeautifulSoup(html, "html.parser")
        extract_and_translate_text_nodes(soup, translator_func, log_callback)
        result = str(soup)
        
        test_soup = BeautifulSoup(result, "html.parser")
        if test_soup.get_text().strip() and len(test_soup.get_text()) > 10:
            return result
        else:
            log_callback("[WARN] Strategy 2 failed, trying fallback")
    except Exception as e:
        log_callback(f"[WARN] Strategy 2 error: {e}")
    
    # Strategy 3: Simple fallback
    try:
        result = translate_html_simple_fallback(html, translator_func, log_callback)
        return result
    except Exception as e:
        log_callback(f"[ERROR] All strategies failed: {e}")
        return html

def translate_json_fields(data, translator_func, log_callback, translated_flags=None, current_path="root"):
    if translated_flags is None:
        translated_flags = set()

    translatable_keys = {
        "text", "question", "title", "alt", "label", "contentName",
        "introduction", "startButtonText", "checkAnswerButton", "submitAnswerButton", 
        "showSolutionButton", "tryAgainButton", "tipsLabel", "scoreBarLabel",
        "tipAvailable", "feedbackAvailable", "readFeedback", "wrongAnswer", 
        "correctAnswer", "shouldCheck", "shouldNotCheck", "noInput",
        "header", "body", "cancelLabel", "confirmLabel", "tip", 
        "chosenFeedback", "notChosenFeedback"
    }

    if isinstance(data, dict):
        for key, value in data.items():
            path = f"{current_path}/{key}"
            if path in translated_flags:
                continue

            # Debug logging for path tracking
            if key in ["answers", "questions"] and isinstance(value, list):
                log_callback(f"[DEBUG] Found {key} array at path: {path}")

            if key in translatable_keys and isinstance(value, str) and value.strip():
                try:
                    original_value = value
                    if "<" in value and ">" in value:
                        translated = translate_html_robust(value, translator_func, log_callback)
                    else:
                        translated = translator_func(value)
                        log_callback(f"[FIELD] {value[:50]}... → {translated[:50]}...")
                    
                    # Additional validation
                    if len(translated.strip()) < 3:
                        log_callback(f"[WARN] Translation too short, keeping original for {key}")
                        translated = original_value
                    
                    data[key] = translated
                    translated_flags.add(path)
                except Exception as e:
                    log_callback(f"[WARN] Couldn't translate {key} at {path}: {e}")
                continue

            # Special handling for answers arrays
            if key == "answers" and isinstance(value, list):
                log_callback(f"[DEBUG] Processing answers array with {len(value)} items")
                for idx, answer in enumerate(value):
                    answer_path = f"{path}[{idx}]"
                    log_callback(f"[DEBUG] Processing answer {idx} at {answer_path}")
                    
                    if isinstance(answer, dict):
                        # Handle the "text" field in answers
                        if "text" in answer:
                            sub_path = f"{answer_path}/text"
                            if sub_path not in translated_flags:
                                orig = answer["text"]
                                if isinstance(orig, str) and orig.strip():
                                    try:
                                        log_callback(f"[DEBUG] Translating answer text: {orig[:60]}...")
                                        if "<" in orig and ">" in orig:
                                            translated = translate_html_robust(orig, translator_func, log_callback)
                                        else:
                                            translated = translator_func(orig)
                                        
                                        if len(translated.strip()) >= 3:
                                            answer["text"] = translated
                                            translated_flags.add(sub_path)
                                            log_callback(f"[ANSWER] {orig[:50]}... → {translated[:50]}...")
                                        else:
                                            log_callback(f"[WARN] Answer translation too short, keeping original")
                                    except Exception as e:
                                        log_callback(f"[ERROR] Translating answer[{idx}] text failed: {orig[:40]}... ({e})")
                                else:
                                    log_callback(f"[DEBUG] Answer text empty or invalid: '{orig}'")
                        else:
                            log_callback(f"[DEBUG] Answer {idx} has no 'text' field")
                        
                        # Also recursively process the rest of the answer object (for nested structures)
                        translate_json_fields(answer, translator_func, log_callback, translated_flags, answer_path)
                    else:
                        log_callback(f"[DEBUG] Answer {idx} is not a dict: {type(answer)}")
                continue

            # Special handling for questions arrays
            if key == "questions" and isinstance(value, list):
                log_callback(f"[DEBUG] Processing questions array with {len(value)} items")
                for idx, question in enumerate(value):
                    question_path = f"{path}[{idx}]"
                    log_callback(f"[DEBUG] Processing question {idx} at {question_path}")
                    translate_json_fields(question, translator_func, log_callback, translated_flags, question_path)
                continue

            # Regular recursive processing for other nested structures
            if isinstance(value, (dict, list)):
                translate_json_fields(value, translator_func, log_callback, translated_flags, path)

    elif isinstance(data, list):
        for idx, item in enumerate(data):
            path = f"{current_path}[{idx}]"
            if isinstance(item, (dict, list)):
                translate_json_fields(item, translator_func, log_callback, translated_flags, path)
            elif isinstance(item, str) and path not in translated_flags:
                try:
                    if "<" in item and ">" in item:
                        translated = translate_html_robust(item, translator_func, log_callback)
                    else:
                        translated = translator_func(item)
                    
                    if len(translated.strip()) >= 3:
                        data[idx] = translated
                        translated_flags.add(path)
                        log_callback(f"[LIST] {item[:30]}... → {translated[:30]}...")
                    else:
                        log_callback(f"[WARN] List item translation too short, keeping original")
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
        self.root.title("H5P Translator (Local AI) - Robust Version")
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