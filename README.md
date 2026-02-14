# Better Reads

**Large-scale book recommendation engine built on 32M+ reader reviews.**

> **This repository is source-available, not open source.**
> Viewing for educational and reference purposes is permitted. All other use is prohibited.
> See [LICENSE](LICENSE) for full terms. Violations will be pursued.

---

## Overview

Better Reads is a production-grade collaborative filtering system trained on one of the largest publicly available book review datasets:

- **26 million** Amazon book reviews
- **6 million** Goodreads reviews
- **32M+ total** reader signals processed

The system generates personalized book recommendations by modeling latent reader preferences at scale.

## Technical Architecture

### Data Pipeline
- Distributed ingestion via **AWS** (S3 + EMR)
- Storage and indexing in **MongoDB**
- Feature engineering and batch processing with **PySpark**

### Recommendation Engine
- **Implicit ALS** (Alternating Least Squares) for collaborative filtering
- Ensemble of **5 models** trained on complementary vector spaces:
  - User-to-user similarity vectors
  - User-to-item interaction vectors
  - Item-to-item co-occurrence vectors
- Model training via **PyTorch**, **Spotlight**, and **Implicit** (Cython + OpenMP optimized)

### NLP Layer
- Topic modeling on review text via **NLTK**
- Sentiment extraction for review quality weighting
- LLM-powered recommendation generation (see `recommend.py`)

## Quick Start

```bash
# LLM-powered recommendations (requires vLLM, Ollama, or OpenAI API key)
python recommend.py "The Great Gatsby"
python recommend.py "Pride and Prejudice" --n 5
```

## Technology Stack

`AWS` `MongoDB` `PySpark` `PyTorch` `Spotlight` `Implicit` `Cython` `OpenMP` `NLTK` `Python`

## Related Work

This research directly informs the recommendation engine in [Readify](https://www.ireadifybooks.com), an AI-powered interactive reading platform for schools and institutions.

---

## Legal Notice

**Copyright (c) 2018-2026 Clarence Stephen. All rights reserved.**

This software is provided under a **Source Available License**. It is **not** open source. You may view this code for educational and reference purposes only. Commercial use, redistribution, modification, derivative works, and incorporation into other products or services are **strictly prohibited** without prior written authorization.

Unauthorized use will result in legal action, including DMCA takedowns, injunctive relief, and claims for damages. See [LICENSE](LICENSE) for complete terms.

For licensing inquiries or institutional partnerships: **clarence@ireadifybooks.com**
