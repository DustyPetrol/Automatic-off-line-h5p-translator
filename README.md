# H5P AI Translator

A robust Python application that translates H5P (HTML5 Package) interactive content files using local AI translation models. Translate educational quizzes, presentations, and interactive content from English to German while preserving functionality and formatting.

## Features

- **Local AI Translation**: Uses Facebook's M2M100 model - no internet required, complete privacy
- **Smart Content Processing**: Handles complex HTML markup within H5P content
- **Robust Fallback System**: Multiple translation strategies ensure nothing gets lost
- **Technical Term Accuracy**: Built-in corrections for common mistranslations in technical content
- **Batch Processing**: Translate entire H5P packages in one go
- **Real-time Logging**: See exactly what's being translated as it happens
- **GUI Interface**: Easy-to-use graphical interface for non-technical users

## Quick Start

### Prerequisites

```bash
# Python 3.8+ required
pip install torch transformers beautifulsoup4 tkinter
```

### Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python translate_h5p_gui.py
   ```

### Usage

1. **Launch** the application - it will automatically download the AI model on first run (this takes a few minutes)
2. **Select** your `.h5p` file using the "Select .h5p File" button
3. **Click** "Translate Now" 
4. **Wait** for translation to complete (watch the log for progress)
5. **Find** your translated file as `filename_translated.h5p`

## What Gets Translated

The system intelligently identifies and translates:

- **Question text** and answer options
- **Instructions** and introductory content  
- **Button labels** (Submit, Check Answer, Try Again, etc.)
- **Feedback messages** (Correct! Incorrect! etc.)
- **Tips and hints**
- **Content titles** and descriptions
- **Alt text** for images

## Technical Details

### Translation Model

- **Default Model**: Facebook M2M100 (1.2B parameters)
- **Alternative Options**: 
  - M2M100-418M (faster, smaller download)
  - Helsinki-NLP opus-mt-en-de (specialized English→German)
- **Languages**: English → German (M2M100 supports 100+ languages with code changes)
- **Processing**: Local inference - no data sent to external servers
- **Performance**: ~30-60 seconds per typical quiz on modern hardware

### Smart Processing Features

#### Text Chunking
Long text is automatically split into manageable chunks to prevent AI model truncation while maintaining context and sentence coherence.

#### HTML Preservation  
Three-tier strategy for handling HTML content:
1. **Element Context**: Translates complete HTML elements for best context
2. **Text Node**: Falls back to individual text node translation  
3. **Simple Fallback**: Emergency plain-text extraction and reconstruction

#### Technical Term Corrections
Built-in dictionary corrects common AI mistranslations:
- "solder" → "löten" (not "Soldat")
- "PCB" → "Leiterplatte" 
- "resistor" → "Widerstand"
- And more...

## Configuration

### Changing Target Language

Edit these lines in `translate_h5p_gui.py`:

```python
SOURCE_LANG = "en"  # English
TARGET_LANG = "de"  # German - change to: "fr", "es", "it", etc.
```

### Model Selection

Choose your speed vs. quality preference:

```python
# MODEL_NAME = "facebook/m2m100_418M"      # Smaller, faster, less accurate
MODEL_NAME = "facebook/m2m100_1.2B"        # Larger, slower, more accurate (recommended)
# MODEL_NAME = "Helsinki-NLP/opus-mt-en-de" # Alternative English->German model
```

### Adding Custom Term Corrections (might not work at the moment)

Extend the corrections dictionary:

```python
TRANSLATION_CORRECTIONS = {
    "your_term": "translation",
    "circuit": "Schaltkreis",
    "voltage": "Spannung",
    # Add your domain-specific terms...
}
```

## System Requirements

### Minimum Requirements
- **OS**: Windows 10, macOS 10.14, or Linux
- **RAM**: 4 GB (8 GB recommended)
- **Storage**: 2 GB free space (for model files)
- **Python**: 3.8 or newer

### Recommended for Best Performance
- **GPU**: NVIDIA GPU with CUDA support (10x faster translation)
- **RAM**: 8+ GB
- **CPU**: Multi-core processor

### Debug Mode

Enable detailed logging by checking "Export translated folder as ZIP" - this creates additional debug files showing the internal JSON structure.

## File Structure

```
your-project/
├── translate_h5p_gui.py           # Main application
├── requirements.txt               # Python dependencies  
├── README.md                     # This file
├── translation-log.txt           # Auto-generated log file
└── temp_h5p/                    # Temporary extraction folder (auto-deleted)
```

## Supported H5P Content Types

Should work with most H5P content types that contain translatable text fields in content.json.

## Acknowledgments

- **Hugging Face Transformers**: For the excellent ML framework
- **Facebook AI**: For the M2M100 translation models  
- **H5P Community**: For creating amazing interactive content tools
- **BeautifulSoup**: For robust HTML parsing
