# ChakrView v0.8.0 Training Corpus Provenance & License Notice

This document details the source attribution, document counts, and licensing terms governing the 5,534 training documents in the Stage C corpus used to train the canonical ChakrView `ChakrMicro` model.

---

## 1. Corpus Overview

- **Corpus Manifest**: `data/manifests/stage_c_manifest.json`
- **Total Documents**: 5,534
- **Reconciliation Status**: 100.0% (5,534 / 5,534 documents mapped to source and license)
- **Tokenized Training Shards**: 27 binary shards (`shard_00000.bin` to `shard_00026.bin`) in `data/tokenized/stage_c/train/`
- **Shard 0 SHA-256**: `606dbc36fa5e66160893118cf94056aa8433df57a62758ae7a525a7eb866d9c6`

---

## 2. Granular Source Attribution & Licensing

| Source Dataset / Corpus | Document Count | Percentage | Applicable License | Attribution / Source Notice |
| :--- | :---: | :---: | :--- | :--- |
| **GSM8K** | 1,805 | 32.6% | **MIT License** | Copyright (c) 2021 OpenAI. Grade School Math 8K corpus. |
| **Simple English Wikipedia** | 1,603 | 29.0% | **CC-BY-SA 4.0** | Wikimedia Foundation contributors. Text available under Creative Commons Attribution-ShareAlike 4.0 International. |
| **Hindi Wikipedia** | 948 | 17.1% | **CC-BY-SA 4.0** | Wikimedia Foundation contributors. Text available under Creative Commons Attribution-ShareAlike 4.0 International. |
| **TheAlgorithms Python** | 539 | 9.7% | **MIT License** | TheAlgorithms Open Source Community. Python algorithmic implementations. |
| **Sanskrit Wikipedia** | 304 | 5.5% | **CC-BY-SA 4.0** | Wikimedia Foundation contributors. Text available under Creative Commons Attribution-ShareAlike 4.0 International. |
| **CPython Documentation** | 304 | 5.5% | **PSF-2.0** | Python Software Foundation. CPython documentation and standard library docstrings. |
| **OpenStax Biology** | 18 | 0.3% | **CC-BY 4.0** | OpenStax, Rice University. High school biology textbook text available under CC-BY 4.0. |
| **L3Cube Marathi/Hindi** | 10 | 0.2% | **CC-BY 4.0** | L3Cube-Pune NLP Research Group. Indian language text snippets. |
| **Open Government Data India** | 3 | 0.1% | **GODL-India** | National Informatics Centre (NIC), Government of India. Open Government Data License. |
| **Total** | **5,534** | **100.0%** | **All Open Licenses** | Clean, non-commercial and commercial open-source data provenance. |

---

## 3. License Notice Requirements

### Creative Commons Attribution-ShareAlike 4.0 (CC-BY-SA 4.0)
Portions of the pretraining data originate from Wikipedia (Simple English, Hindi, and Sanskrit editions). In accordance with Section 3 of CC-BY-SA 4.0:
- Attribution is provided to the Wikimedia Foundation contributors.
- If you remix, transform, or build upon the material contained in these documents, you must distribute your contributions under the same license as the original.

### Python Software Foundation License (PSF-2.0)
Portions of code documentation originate from Python Software Foundation materials:
- Copyright (c) 2001-2026 Python Software Foundation; All Rights Reserved.

### MIT License
Portions originating from GSM8K and TheAlgorithms:
- Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files...
