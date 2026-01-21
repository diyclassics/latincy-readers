# LatinCy Readers

Corpus readers for Latin texts with [LatinCy](https://github.com/diyclassics/latincy)/spaCy integration.

Version 1.0.0a1; Python 3.10+; LatinCy 3.8.0+

## Installation

```bash
# Install from GitHub
pip install git+https://github.com/diyclassics/latincy-readers.git

# With LatinCy model included
pip install "latincy-readers[models] @ git+https://github.com/diyclassics/latincy-readers.git"

# For development (editable install)
git clone https://github.com/diyclassics/latincy-readers.git
cd latincy-readers
pip install -e ".[dev]"
```

## Quick Start

```python
from latincyreaders import TesseraeReader, AnnotationLevel

# Initialize reader
reader = TesseraeReader("/path/to/tesserae/corpus")

# Iterate over documents as spaCy Docs
for doc in reader.docs():
    print(f"{doc._.fileid}: {len(list(doc.sents))} sentences")

# Search for sentences containing specific forms
for result in reader.find_sents(forms=["Caesar", "Caesarem"]):
    print(f"{result['citation']}: {result['sentence']}")

# Get raw text (no NLP processing)
for text in reader.texts():
    print(text[:100])
```

## Readers

| Reader | Format | Description |
|--------|--------|-------------|
| `TesseraeReader` | `.tess` | CLTK Tesserae corpus format |
| `PlaintextReader` | `.txt` | Plain text files |
| `LatinLibraryReader` | `.txt` | Latin Library-style plaintext |
| `TEIReader` | `.xml` | TEI-XML documents |
| `PerseusReader` | `.xml` | Perseus Digital Library TEI |
| `CamenaReader` | `.xml` | CAMENA Neo-Latin corpus |
| `TxtdownReader` | `.txtd` | Txtdown format with citations |

## Core API

All readers provide a consistent interface:

```python
reader.fileids()              # List available files
reader.texts(fileids=...)     # Raw text strings (generator)
reader.docs(fileids=...)      # spaCy Doc objects (generator)
reader.sents(fileids=...)     # Sentence spans (generator)
reader.tokens(fileids=...)    # Token objects (generator)
reader.metadata(fileids=...)  # File metadata (generator)
reader.describe()             # Corpus statistics
```

### Search API

```python
# Fast regex search (no NLP)
reader.search(pattern=r"\bbell\w+")

# Form-based sentence search
reader.find_sents(forms=["amor", "amoris"])

# Lemma-based search (requires NLP)
reader.find_sents(lemmas=["amo", "bellum"])
```

### Annotation Levels

Control NLP processing overhead:

```python
from latincyreaders import AnnotationLevel

# No NLP - fastest, returns raw strings
reader.texts()

# Tokenization only
reader.docs(annotation_level=AnnotationLevel.TOKENIZE)

# Basic: tokenization + sentence boundaries (default)
reader.docs(annotation_level=AnnotationLevel.BASIC)

# Full pipeline: POS, lemma, morphology, NER
reader.docs(annotation_level=AnnotationLevel.FULL)
```

## Corpora Supported

- [CLTK Tesserae Latin Corpus](https://github.com/cltk/lat_text_tesserae)
- [CLTK Tesserae Greek Corpus](https://github.com/cltk/grc_text_tesserae)
- [Perseus Digital Library TEI](https://www.perseus.tufts.edu/)
- [Latin Library](https://www.thelatinlibrary.com/)
- [CAMENA Neo-Latin](https://github.com/nevenjovanovic/camena-neolatinlit)
- [Open Greek & Latin CSEL](https://github.com/OpenGreekAndLatin/csel-dev)
- Any plaintext or TEI-XML collection

## CLI Tools

Search tools in `cli/`:

```bash
# Wordform search
python cli/token_search.py --forms Caesar Caesarem --limit 100

# Lemma search
python cli/lemma_search.py --lemmas bellum pax --fileids "cicero.*"
```

---

*Developed by [Patrick J. Burns](http://github.com/diyclassics)*
