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
# MODEL_NAME = "facebook/m2m100_418M"      # Smaller, faster, less accurate
MODEL_NAME = "facebook/m2m100_1.2B"        # Larger, slower, more accurate
# MODEL_NAME = "Helsinki-NLP/opus-mt-en-de" # Alternative English->German model

# Language configurations
SUPPORTED_LANGUAGES = {
    "en": "English",
    "de": "German", 
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean"
}

SOURCE_LANG = "en"
TARGET_LANG = "de"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME)
model = M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
tokenizer.src_lang = SOURCE_LANG

def translate_local_ai(text, target_lang=TARGET_LANG):
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
                    translated_parts.append(translate_single_chunk(current_chunk.strip(), target_lang))
                    current_chunk = full_sentence
                else:
                    # Single sentence too long, translate as-is
                    translated_parts.append(translate_single_chunk(full_sentence.strip(), target_lang))
            else:
                current_chunk += full_sentence
        
        if current_chunk:
            translated_parts.append(translate_single_chunk(current_chunk.strip(), target_lang))
        
        return " ".join(translated_parts)
    else:
        return translate_single_chunk(text, target_lang)

# === Translation Corrections Dictionary ===
TRANSLATION_CORRECTIONS = {
    # English -> German corrections for technical terms
    "solder": "löten",
    "soldering": "löten", 
    "soldered": "gelötet",
    "soldier": "Soldat",  # Keep this for actual military context
    "soldator": "löten",  # Fix common mistranslation
    # Add more technical terms as needed
    "flux": "Flussmittel",
    "PCB": "Leiterplatte",
    "resistor": "Widerstand",
    "capacitor": "Kondensator",
    "transistor": "Transistor",
    "LED": "LED",
    "breadboard": "Steckbrett",
}

def apply_translation_corrections(text, corrections_dict):
    """Apply manual corrections to translated text"""
    corrected = text
    for english_term, german_term in corrections_dict.items():
        # Case-insensitive replacement, preserving original case pattern
        import re
        pattern = re.compile(re.escape(english_term), re.IGNORECASE)
        
        def replace_match(match):
            original = match.group()
            if original.isupper():
                return german_term.upper()
            elif original.istitle():
                return german_term.capitalize()
            else:
                return german_term.lower()
        
        corrected = pattern.sub(replace_match, corrected)
    
    return corrected

def translate_single_chunk(text, target_lang=TARGET_LANG):
    """Translate a single chunk with error handling and corrections"""
    try:
        log_text = text[:50] + "..." if len(text) > 50 else text
        print(f"[TRANSLATE-DEBUG] Input: '{log_text}'")
        
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
        generated = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
            max_length=400,  # Increased max length
            num_beams=3,     # Better quality
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id
        )
        result = tokenizer.decode(generated[0], skip_special_tokens=True)
        
        # Apply manual corrections
        corrected_result = apply_translation_corrections(result, TRANSLATION_CORRECTIONS)
        
        log_result = result[:50] + "..." if len(result) > 50 else result
        log_corrected = corrected_result[:50] + "..." if len(corrected_result) > 50 else corrected_result
        
        print(f"[TRANSLATE-DEBUG] Raw output: '{log_result}'")
        if corrected_result != result:
            print(f"[TRANSLATE-DEBUG] After corrections: '{log_corrected}'")
        
        # Verify result is reasonable
        if len(corrected_result.strip()) < 3:
            print(f"[TRANSLATE-DEBUG] Result too short, using original")
            return text  # Fallback to original
        
        return corrected_result
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

def translate_json_fields(data, translator_func, log_callback, target_lang, translated_flags=None, current_path="root"):
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
                        translated = translate_html_robust(value, lambda x: translator_func(x, target_lang), log_callback)
                    else:
                        translated = translator_func(value, target_lang)
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
                                            translated = translate_html_robust(orig, lambda x: translator_func(x, target_lang), log_callback)
                                        else:
                                            translated = translator_func(orig, target_lang)
                                        
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
                        translate_json_fields(answer, translator_func, log_callback, target_lang, translated_flags, answer_path)
                    else:
                        log_callback(f"[DEBUG] Answer {idx} is not a dict: {type(answer)}")
                continue

            # Special handling for questions arrays
            if key == "questions" and isinstance(value, list):
                log_callback(f"[DEBUG] Processing questions array with {len(value)} items")
                for idx, question in enumerate(value):
                    question_path = f"{path}[{idx}]"
                    log_callback(f"[DEBUG] Processing question {idx} at {question_path}")
                    translate_json_fields(question, translator_func, log_callback, target_lang, translated_flags, question_path)
                continue

            # Regular recursive processing for other nested structures
            if isinstance(value, (dict, list)):
                translate_json_fields(value, translator_func, log_callback, target_lang, translated_flags, path)

    elif isinstance(data, list):
        for idx, item in enumerate(data):
            path = f"{current_path}[{idx}]"
            if isinstance(item, (dict, list)):
                translate_json_fields(item, translator_func, log_callback, target_lang, translated_flags, path)
            elif isinstance(item, str) and path not in translated_flags:
                try:
                    if "<" in item and ">" in item:
                        translated = translate_html_robust(item, lambda x: translator_func(x, target_lang), log_callback)
                    else:
                        translated = translator_func(item, target_lang)
                    
                    if len(translated.strip()) >= 3:
                        data[idx] = translated
                        translated_flags.add(path)
                        log_callback(f"[LIST] {item[:30]}... → {translated[:30]}...")
                    else:
                        log_callback(f"[WARN] List item translation too short, keeping original")
                except Exception as e:
                    log_callback(f"[WARN] Couldn't translate list item at {path}: {e}")

def fix_moodle_h5p_for_lumi(input_h5p, log_callback):
    """Fix Moodle H5P files for Lumi compatibility by removing missing library references"""
    temp_dir = "temp_lumi_fix"
    
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    # Extract H5P file
    with zipfile.ZipFile(input_h5p, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    log_callback("[LUMI-FIX] Checking for missing library references...")
    
    # Load main H5P metadata
    h5p_json_path = os.path.join(temp_dir, "h5p.json")
    content_json_path = os.path.join(temp_dir, "content", "content.json")
    
    fixed_anything = False
    
    # Fix h5p.json - remove references to missing libraries
    if os.path.exists(h5p_json_path):
        with open(h5p_json_path, 'r', encoding='utf-8') as f:
            h5p_json = json.load(f)
        
        # Check what library folders actually exist
        existing_libraries = set()
        for item in os.listdir(temp_dir):
            if os.path.isdir(os.path.join(temp_dir, item)) and item not in ['content', 'temp_lumi_fix']:
                existing_libraries.add(item)
        
        log_callback(f"[LUMI-FIX] Found library folders: {existing_libraries}")
        
        # Remove missing dependencies
        if "preloadedDependencies" in h5p_json:
            original_deps = h5p_json["preloadedDependencies"][:]
            h5p_json["preloadedDependencies"] = []
            
            for dep in original_deps:
                lib_folder = f"{dep['machineName']}-{dep['majorVersion']}.{dep['minorVersion']}"
                if lib_folder in existing_libraries:
                    h5p_json["preloadedDependencies"].append(dep)
                    log_callback(f"[LUMI-FIX] Kept dependency: {lib_folder}")
                else:
                    log_callback(f"[LUMI-FIX] Removed missing dependency: {lib_folder}")
                    fixed_anything = True
        
        # Same for editor dependencies
        if "editorDependencies" in h5p_json:
            original_deps = h5p_json["editorDependencies"][:]
            h5p_json["editorDependencies"] = []
            
            for dep in original_deps:
                lib_folder = f"{dep['machineName']}-{dep['majorVersion']}.{dep['minorVersion']}"
                if lib_folder in existing_libraries:
                    h5p_json["editorDependencies"].append(dep)
                else:
                    log_callback(f"[LUMI-FIX] Removed missing editor dependency: {lib_folder}")
                    fixed_anything = True
        
        # Save fixed h5p.json
        with open(h5p_json_path, 'w', encoding='utf-8') as f:
            json.dump(h5p_json, f, ensure_ascii=False, indent=2)
    
    # Fix content.json - remove showWhen and other editor-specific fields
    if os.path.exists(content_json_path):
        with open(content_json_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
        
        def remove_editor_fields_recursively(obj, path=""):
            nonlocal fixed_anything
            if isinstance(obj, dict):
                keys_to_remove = []
                for key, value in obj.items():
                    current_path = f"{path}/{key}" if path else key
                    # Remove editor-specific fields
                    if key in ["showWhen", "widget", "importance", "description"]:
                        keys_to_remove.append(key)
                        log_callback(f"[LUMI-FIX] Removing editor field: {current_path}")
                        fixed_anything = True
                    else:
                        remove_editor_fields_recursively(value, current_path)
                
                for key in keys_to_remove:
                    del obj[key]
                    
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    remove_editor_fields_recursively(item, f"{path}[{i}]")
        
        remove_editor_fields_recursively(content)
        
        # Save fixed content.json
        with open(content_json_path, 'w', encoding='utf-8') as f:
            json.dump(content, f, ensure_ascii=False, indent=2)
    
    if fixed_anything:
        # Create fixed H5P file
        fixed_filename = input_h5p.replace('.h5p', '_lumi_fixed.h5p')
        with zipfile.ZipFile(fixed_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname)
        
        log_callback(f"[✅] Lumi compatibility fixes applied: {fixed_filename}")
        shutil.rmtree(temp_dir)
        return fixed_filename
    else:
        log_callback("[LUMI-FIX] No fixes needed - file should work in Lumi as-is")
        shutil.rmtree(temp_dir)
        return input_h5p

def translate_h5p(input_h5p, output_h5p, log_callback, source_lang, target_lang, export_raw=False, fix_for_lumi=False):
    # Update tokenizer source language
    global tokenizer
    tokenizer.src_lang = source_lang
    
    # Apply Lumi compatibility fixes first if requested
    working_file = input_h5p
    if fix_for_lumi:
        log_callback("[INFO] Applying Lumi compatibility fixes...")
        working_file = fix_moodle_h5p_for_lumi(input_h5p, log_callback)
    
    temp_dir = "temp_h5p"
    content_json_path = os.path.join(temp_dir, "content", "content.json")

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    with zipfile.ZipFile(working_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    log_callback("[OK] Extracted H5P")

    with open(content_json_path, 'r', encoding='utf-8') as f:
        content = json.load(f)

    log_callback("[INFO] Starting translation...")
    translate_json_fields(content, translate_local_ai, log_callback, target_lang)

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
    
    # Clean up temporary fixed file if we created one
    if fix_for_lumi and working_file != input_h5p and os.path.exists(working_file):
        os.remove(working_file)
        log_callback("[INFO] Cleaned up temporary Lumi-fixed file")

# === GUI ===
class TranslatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("H5P Translator (Local AI) - Enhanced Version")
        self.root.geometry("900x750")

        self.file_path = tk.StringVar()
        self.output_folder = tk.StringVar()
        self.source_lang = tk.StringVar(value="en")
        self.target_lang = tk.StringVar(value="de")
        self.export_raw = tk.BooleanVar()
        self.fix_for_lumi = tk.BooleanVar()
        self.log_file_path = "translation-log.txt"

        self.setup_ui()

    def setup_ui(self):
    # Initialize language display variables first
        self.source_lang_display = tk.StringVar()
        self.target_lang_display = tk.StringVar()
    
    # File selection
        file_frame = tk.Frame(self.root)
        file_frame.pack(pady=5, fill="x", padx=10)
    
        tk.Label(file_frame, text="Input H5P File:", font=("Arial", 10, "bold")).pack(anchor="w")
        input_frame = tk.Frame(file_frame)
        input_frame.pack(fill="x", pady=(2, 5))
    
        tk.Entry(input_frame, textvariable=self.file_path, width=80).pack(side="left", fill="x", expand=True)
        tk.Button(input_frame, text="Browse", command=self.select_file).pack(side="right", padx=(5, 0))

        # Output folder selection
        tk.Label(file_frame, text="Output Folder:", font=("Arial", 10, "bold")).pack(anchor="w")
        output_frame = tk.Frame(file_frame)
        output_frame.pack(fill="x", pady=(2, 10))
    
        tk.Entry(output_frame, textvariable=self.output_folder, width=80).pack(side="left", fill="x", expand=True)
        tk.Button(output_frame, text="Browse", command=self.select_output_folder).pack(side="right", padx=(5, 0))

        # Language selection
        lang_frame = tk.Frame(self.root)
        lang_frame.pack(pady=5, fill="x", padx=10)
    
        tk.Label(lang_frame, text="Language Settings:", font=("Arial", 10, "bold")).pack(anchor="w")
    
        lang_controls = tk.Frame(lang_frame)
        lang_controls.pack(fill="x", pady=(5, 10))
    
    # Source language
        source_frame = tk.Frame(lang_controls)
        source_frame.pack(side="left", padx=(0, 20))
        tk.Label(source_frame, text="From:").pack()
        source_combo = tk.OptionMenu(source_frame, self.source_lang, *SUPPORTED_LANGUAGES.keys())
        source_combo.config(width=8)
        source_combo.pack()
        tk.Label(source_frame, textvariable=self.source_lang_display, font=("Arial", 8)).pack()
    
    # Arrow
        tk.Label(lang_controls, text="→", font=("Arial", 16)).pack(side="left")
    
    # Target language  
        target_frame = tk.Frame(lang_controls)
        target_frame.pack(side="left", padx=(20, 0))
        tk.Label(target_frame, text="To:").pack()
        target_combo = tk.OptionMenu(target_frame, self.target_lang, *SUPPORTED_LANGUAGES.keys())
        target_combo.config(width=8)
        target_combo.pack()
        tk.Label(target_frame, textvariable=self.target_lang_display, font=("Arial", 8)).pack()

    # Options
        options_frame = tk.Frame(self.root)
        options_frame.pack(pady=5, fill="x", padx=10)
    
        tk.Label(options_frame, text="Options:", font=("Arial", 10, "bold")).pack(anchor="w")
        tk.Checkbutton(options_frame, text="Fix for Lumi compatibility (removes missing library references)", 
                  variable=self.fix_for_lumi).pack(anchor="w", pady=2)
        tk.Checkbutton(options_frame, text="Export translated folder as ZIP (for debugging)", 
                  variable=self.export_raw).pack(anchor="w", pady=2)

    # Translate button
        tk.Button(self.root, text="🔄 Translate Now", command=self.start_translation, 
                font=("Arial", 12, "bold"), bg="#4CAF50", fg="white", pady=10).pack(pady=15)

    # Log area
        log_frame = tk.Frame(self.root)
        log_frame.pack(pady=5, fill="both", expand=True, padx=10)
    
        tk.Label(log_frame, text="Translation Log:", font=("Arial", 10, "bold")).pack(anchor="w")
        self.log = scrolledtext.ScrolledText(log_frame, height=20, width=100)
        self.log.pack(fill="both", expand=True, pady=(5, 0))

    # Status
        self.status = tk.Label(self.root, text="Ready", fg="blue", font=("Arial", 10, "bold"))
        self.status.pack(pady=(5, 10))

    # Set up initial language displays and bind change events
        self.update_language_displays()
        self.source_lang.trace("w", lambda *args: self.update_language_displays())
        self.target_lang.trace("w", lambda *args: self.update_language_displays())

    def update_language_displays(self):
        source_code = self.source_lang.get()
        target_code = self.target_lang.get()
        self.source_lang_display.set(SUPPORTED_LANGUAGES.get(source_code, source_code))
        self.target_lang_display.set(SUPPORTED_LANGUAGES.get(target_code, target_code))

    def select_output_folder(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder.set(folder)

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

        # Determine output file path
        if self.output_folder.get():
            filename = os.path.basename(file)
            name_without_ext = filename.replace(".h5p", "")
            source_lang_name = SUPPORTED_LANGUAGES.get(self.source_lang.get(), self.source_lang.get())
            target_lang_name = SUPPORTED_LANGUAGES.get(self.target_lang.get(), self.target_lang.get())
            output_file = os.path.join(self.output_folder.get(), f"{name_without_ext}_{source_lang_name}_to_{target_lang_name}.h5p")
        else:
            output_file = file.replace(".h5p", "_translated.h5p")

        if os.path.exists(output_file):
            overwrite = messagebox.askyesno("Overwrite?", f"{os.path.basename(output_file)} already exists. Overwrite?")
            if not overwrite:
                self.status.config(text="Cancelled", fg="orange")
                return

        # Validate language selection
        if self.source_lang.get() == self.target_lang.get():
            messagebox.showwarning("Warning", "Source and target languages are the same!")
            return

        self.status.config(text="Translating...", fg="black")
        self.log.delete("1.0", tk.END)
        with open(self.log_file_path, "w", encoding="utf-8") as f:
            f.write(f"Translating file: {file}\n")
            f.write(f"From {SUPPORTED_LANGUAGES.get(self.source_lang.get())} to {SUPPORTED_LANGUAGES.get(self.target_lang.get())}\n")
            f.write(f"Output: {output_file}\n\n")

        threading.Thread(target=self.run_translation, args=(file, output_file, self.source_lang.get(), 
                                                           self.target_lang.get(), self.export_raw.get(), 
                                                           self.fix_for_lumi.get())).start()

    def run_translation(self, input_file, output_file, source_lang, target_lang, export_raw, fix_for_lumi):
        try:
            translate_h5p(input_file, output_file, self.log_msg, source_lang, target_lang, export_raw, fix_for_lumi)
            self.status.config(text="✅ Translation complete", fg="green")
        except Exception as e:
            self.log_msg(f"[ERROR] {e}")
            self.status.config(text="❌ Error occurred", fg="red")

# === Run GUI App ===
if __name__ == "__main__":
    root = tk.Tk()
    app = TranslatorGUI(root)
    root.mainloop()