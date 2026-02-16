# Base de données CV (CVtheque)

This document describes the **database** of CVs used for OCR and parsing benchmarks. The metadata is stored in `metadata.csv`.

---

## Columns in `metadata.csv`

| Column | Meaning | 0 | 1 |
|--------|---------|---|---|
| **filename** | CV file name | — | e.g. cv001.docx, cv016.pdf |
| **extension** | File format | — | docx, pdf, png, jpg |
| **is_scanned** | Document type | **0 = No** (digital/native) | **1 = Yes** (scanned or image) |
| **language** | Main language | — | fr, en, ar |
| **layout_type** | Layout style | — | mono, multi, infographic |
| **num_columns** | Column count | — | 1 or 2 |
| **has_tables** | Contains tables | **0 = No** | **1 = Yes** |
| **has_icons** | Contains icons | **0 = No** | **1 = Yes** |
| **has_graphics** | Contains graphics/figures | **0 = No** | **1 = Yes** |
| **is_rtl** | Right-to-left (Arabic) | **0 = No** (LTR) | **1 = Yes** (RTL / Arabic) |

---

## Description of our database

### Size

- **Total number of CVs:** 16 (cv001 through cv016).

### By format (extension)

| Format | Count | CVs |
|--------|-------|-----|
| **pdf** | 9 | cv005, cv007, cv009, cv010, cv011, cv012, cv013, cv014, cv016 |
| **docx** | 4 | cv001, cv002, cv003, cv015 |
| **png** | 2 | cv004, cv006 |
| **jpg** | 1 | cv008 |

### Scanned vs native (is_scanned)

| Value | Meaning | Count |
|-------|---------|-------|
| **0** | No — digital/native (editable source) | **10** CVs |
| **1** | Yes — scanned or image | **6** CVs |

### RTL / Arabic (is_rtl)

| Value | Meaning | Count |
|-------|---------|-------|
| **0** | No — left-to-right (French, English) | **12** CVs |
| **1** | Yes — right-to-left (Arabic) | **4** CVs (cv004, cv006, cv010, cv012) |

### Tables (has_tables)

| Value | Count |
|-------|-------|
| **0** (no tables) | 14 CVs |
| **1** (has tables) | **2** CVs (cv015, cv016) |

### Icons (has_icons)

| Value | Count |
|-------|-------|
| **0** (no icons) | 9 CVs |
| **1** (has icons) | **7** CVs (cv006, cv008, cv009, cv011, cv012, cv013, cv014) |

### Graphics (has_graphics)

| Value | Count |
|-------|-------|
| **0** (no graphics) | 8 CVs |
| **1** (has graphics) | **8** CVs (cv003, cv006, cv008, cv009, cv011, cv012, cv013, cv014) |

### By language

| Language | Count | CVs |
|----------|-------|-----|
| **fr** (French) | 8 | cv001, cv002, cv003, cv008, cv011, cv013, cv014, cv015 |
| **en** (English) | 4 | cv005, cv007, cv009, cv016 |
| **ar** (Arabic) | 4 | cv004, cv006, cv010, cv012 |

### By layout type

| Layout | Count | CVs |
|--------|-------|-----|
| **mono** (single column) | 6 | cv001, cv004, cv005, cv010, cv015, cv016 |
| **multi** (multi-column) | 7 | cv002, cv007, cv008, cv011, cv012, cv013, cv014 |
| **infographic** | 2 | cv003, cv009 |

### By number of columns (num_columns)

| Columns | Count |
|---------|-------|
| 1 | 6 CVs |
| 2 | 10 CVs |

---

## Transcription files

Individual text files `cv001.txt`, `cv002.txt`, … one per CV. Each file contains the transcription for that CV. Current set: cv001–cv013, cv015, cv016, cv017 (16 files; cv014 has no corresponding .txt in this set).

---

## Folder layout

```
database/
├── metadata.csv               # Index: filename, language, layout, etc.
├── DATABASE.md                # This file (description of the database)
├── cv001.txt … cv017.txt      # One .txt per CV (transcription; cv014 missing)
└── cv016.pdf                  # (and other source files when present)
```

The metadata was created by inspecting the files (format, language, layout, etc.). The `cvXXX.txt` files hold the transcriptions (from OCR or native text extraction).
