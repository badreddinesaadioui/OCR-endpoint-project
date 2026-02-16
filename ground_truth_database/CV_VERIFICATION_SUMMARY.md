# CV verification summary (source → .txt & .json)

Same process as cv001/cv002/cv003: ensure all content from the source document is present in `parsed/cvXXX.txt` and `json_parsed/cvXXX.json`, and fix any missing or truncated content.

## Already done before this run
- **cv001** (DOCX): JSON – completed 3 truncated experience descriptions (Groupe BPCE, DIFMEDI, TNP Consultants). TXT already complete.
- **cv002** (DOCX): TXT – added missing footer `©AZURIUS – Modeles-de-cv,com`. JSON already complete.
- **cv003** (DOCX): JSON – added “(1 semaine)” to EDF description; added `interests` (DIPLOMAS Y HOBBIES / Formaciones / Hobbies). TXT already complete.

## Done in this run

| CV   | Source | Changes |
|------|--------|--------|
| **cv004** | PNG (ar) | JSON: extended `about` with the two missing sentences (شغف، دعم نفسي، علاج سلوكي، جودة الحياة). TXT already complete. |
| **cv005** | PDF (en) | JSON: full `about`; all four experience descriptions expanded to full bullets from TXT. TXT already complete. (PDF is scanned – no text extraction.) |
| **cv006** | PNG (ar) | JSON: full `about` (full lorem paragraph); all four experience descriptions set to full lorem text. TXT already complete. |
| **cv007** | PDF (en) | JSON: completed `about` with “Bringing excellent communication…”; expanded all four experience descriptions to full text from TXT. TXT already complete. |
| **cv008** | JPG (fr) | No changes. TXT and JSON already match and complete. |
| **cv009** | PDF (en) | No changes. Infographic CV; main content present in JSON. |
| **cv010** | PDF (ar) | No changes. TXT in Arabic, JSON in English; structure and content aligned. |
| **cv011** | PDF (fr) | No changes. Descriptions and sections already complete. |
| **cv012** | PDF (ar) | No changes. TXT Arabic, JSON English; structure aligned. |
| **cv013** | PDF (fr) | No changes. Same structure as cv008 (Olivia Wilson); TXT and JSON complete. |
| **cv014** | — | No parsed files found in `parsed/` or `json_parsed/` for cv014. |
| **cv015** | DOCX (fr) | No changes. **Note:** `cv015.docx` content is “Jean-Pierre Martin” (French); `parsed/cv015.txt` and `json_parsed/cv015.json` are “Camilla Martine”. Possible file mix-up – parsed files left as is. |
| **cv016** | PDF (en) | Already fixed earlier (French degree terms translated to English in TXT and JSON). |

## Summary
- **Updated:** cv004, cv005, cv006, cv007 (JSON only).
- **Checked, no changes:** cv008, cv009, cv010, cv011, cv012, cv013, cv016.
- **cv014:** no ground truth files to verify.
- **cv015:** source (DOCX) and parsed (Camilla Martine) content mismatch; parsed files unchanged.
