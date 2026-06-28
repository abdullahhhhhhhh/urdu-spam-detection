# Urdu SMS Spam Detection using Multilingual Transformers

**Course:** Natural Language Processing (NLP)  
**Team:** Abdullah Kashif (F2023266531), Abdul Mannan (F2023266541), Taha Ahmad Qureshi (F2023266528)

---

## Problem Statement

Pakistan receives millions of Urdu-script spam SMS messages daily, yet no automated detector exists for Urdu. Standard English spam filters fail because Urdu uses a completely different script (Arabic-derived Nastaliq, right-to-left). This project builds the first transformer-based Urdu SMS spam classifier.

## Solution

We compare three approaches:

| Model | Type | Key Idea |
|-------|------|----------|
| **Baseline (NB + SVM)** | Classical ML | TF-IDF features + Naive Bayes / SVM |
| **LSTM** | Deep Learning | Character-level bidirectional LSTM |
| **mBERT** | Transformer | Fine-tuned Multilingual BERT (pre-trained on Urdu) |

## How to Run

### Option 1 — Google Colab (Recommended)
Open `Urdu_Spam_Detection_Colab.ipynb` in Google Colab and run all cells.

### Option 2 — Local Setup

```bash
pip install -r requirements.txt

# Run baseline models (NB + SVM + LSTM)
python src/baseline_model.py

# Run mBERT transformer model
python src/transformer_model.py

# Compare all models
python src/compare_models.py
```

## Project Structure

```
urdu-spam-detection/
├── data/
│   └── sample_data.py          # 100 synthetic Urdu SMS (50 spam, 50 ham)
├── src/
│   ├── baseline_model.py       # Naive Bayes, SVM, LSTM baselines
│   ├── transformer_model.py    # mBERT fine-tuning
│   └── compare_models.py       # Side-by-side comparison — START HERE
├── results/                    # Auto-generated evaluation outputs
├── Urdu_Spam_Detection_Colab.ipynb
├── requirements.txt
└── README.md
```

## Research Gap Addressed

- Papers [1,2] addressed Urdu spam but used outdated ML (LSTM, SVM)
- Paper [4] showed transformers reach 98.5% on English SMS — never tested on Urdu
- **This project**: first application of multilingual transformer (mBERT) to Urdu SMS spam

## Expected Results

| Model | Expected Accuracy |
|-------|------------------|
| Naive Bayes | ~75-80% |
| SVM | ~80-85% |
| LSTM | ~85-90% |
| mBERT (ours) | ~92-97% |

## Key Concepts (for Viva)

- **Transfer Learning**: mBERT pre-trained on 104 languages including Urdu — adapts existing knowledge
- **Fine-tuning**: Only output layer trains from scratch; BERT weights adapt with low learning rate (2e-5)
- **Low-resource NLP**: Urdu has limited labeled data; pre-training bridges this gap
- **mBERT vs XLM-R**: XLM-R trained on more Urdu data, typically 1-2% better on Urdu tasks
