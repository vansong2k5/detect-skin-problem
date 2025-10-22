# Hybrid AI Skin Analyzer

A research-oriented reference implementation of a **Hybrid AI Skin Analyzer** that combines
computer vision embeddings, lightweight classical machine learning, triage rules, and large
language model (LLM) reasoning. The goal is to maximise dermatology safety by pairing
pattern-recognition from CNNs with transparent reasoning layers.

> ⚠️ **Medical disclaimer**: This project is for research and educational purposes only.
> It must not be used as a substitute for professional medical advice, diagnosis, or
> treatment.

## Project layout

```
.
├── README.md
├── requirements.txt
├── data/
│   ├── train/
│   └── val/
├── artifacts/
│   └── .gitignore
└── src/
    └── skin_mvp/
        ├── __init__.py
        ├── features_vision.py
        ├── infer_pipeline.py
        ├── prompt_builder.py
        ├── rules.py
        └── train_clf.py
```

* `data/` – place curated image datasets under class-labelled folders (e.g. `eczema/`).
* `artifacts/` – cached embeddings, training caches, and trained classifiers.
* `src/skin_mvp/` – Python package implementing the full hybrid pipeline.

## Quickstart

1. **Create an environment**

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Prepare training data**

   Organise labelled images as described in `data/train/<class_name>/*.jpg`.
   Recommended starter taxonomy: `eczema`, `impetigo`, `cellulitis`, `tinea`,
   `urticaria`, `acne`, `zoster`.

3. **Extract embeddings and train the classifier**

   ```bash
   python -m src.skin_mvp.train_clf data/train \
       --cache-dir artifacts \
       --device cpu  # or cuda
   ```

   This command creates a cached feature store (`train_cache.pkl`) and trains a
   balanced multinomial logistic regression classifier. The resulting model bundle
   is saved as `artifacts/vision_clf.joblib` containing the scaler, classifier, and
   evaluation metrics.

4. **Run inference**

   ```python
   from src.skin_mvp import VisionInferencePipeline

   pipeline = VisionInferencePipeline("artifacts/vision_clf.joblib", device="cpu")
   result = pipeline.predict_from_path("/path/to/test_image.jpg")
   print(result)
   ```

5. **Combine with symptoms and build LLM prompt**

   ```python
   from src.skin_mvp import Symptoms, integrate_assessment, build_prompt

   symptoms = Symptoms(
       age=28,
       days=3,
       itch=True,
       pain=False,
       rapid_spread=False,
   )

   vision_summary = pipeline.predict_from_path("/path/to/test_image.jpg").to_dict()
   prompt = build_prompt(symptoms, vision_summary)
   print(prompt)
   ```

   Use your preferred LLM SDK to obtain a response and feed it into
   `integrate_assessment` to apply the safety gate.

## Safety gate logic

The final triage recommendation is decided by:

1. Hard **red-flag rules** (ocular involvement, mucosal lesions, airway compromise,
   etc.).
2. Structured heuristics (`level_from_rules`) using symptoms and image signals.
3. Optional LLM response – only allowed to **escalate** severity beyond the rule-based
   level. If the vision confidence is below 0.45 the rules are prioritised.

Every inference should log the intermediate outputs to support auditing and future
human-in-the-loop refinement.

## Dataset strategy

* Start with a small, well-curated labelled seed set per class.
* Use the cached embeddings to bootstrap pseudo-labelling or active-learning loops.
* Re-train frequently and monitor drift metrics when new data sources appear.

## Contributing

Contributions are welcome via pull requests. Please add tests or example notebooks where
possible, document non-obvious decisions, and respect the medical safety disclaimer.

## License

This project is shared under the MIT License. See `LICENSE` (to be added by adopters) for
usage terms.
