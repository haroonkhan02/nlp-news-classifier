# BERT News Article Classifier

End-to-end NLP pipeline for multi-class news article classification using fine-tuned BERT. Classifies articles into 4 categories: **World**, **Sports**, **Business**, and **Sci/Tech**.

## Results

| Metric | Value |
|--------|-------|
| Val Macro F1 | **0.9412** |
| Val Accuracy | **0.9438** |
| Val Loss | 0.1847 |
| Best Epoch | 4/5 (early stopping, patience=2) |

### Per-Class Performance

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| World | 0.952 | 0.948 | 0.950 |
| Sports | 0.981 | 0.977 | 0.979 |
| Business | 0.919 | 0.924 | 0.921 |
| Sci/Tech | 0.917 | 0.914 | 0.916 |

**Known failure mode:** 38% of Sci/Tech→Business misclassifications involve articles about tech company earnings/IPOs (e.g., financial vocabulary triggers Business prediction for tech-finance crossover articles). The `--multi_layer_head` option improves this boundary by ~1.8% F1.

## Architecture

- **Base model:** `bert-base-uncased` (110M params)
- **Strategy:** Freeze embedding layers, fine-tune upper Transformer layers + classification head
- **Pooling:** CLS token (standard) or multi-layer attention head (optional, for boundary cases)
- **Loss:** Cross-entropy with label smoothing (ε=0.1) + AdamW weight decay (λ=0.01)
- **Optimization:** AdamW, lr=2e-5, linear warmup (10%) + linear decay

### Key Design Decisions

1. **Frozen embeddings:** Reduces trainable params by ~25%, enables efficient training on single GPU (~45 min on RTX 3090)
2. **Pre-tokenized caching:** Offline tokenization cached as memory-mapped numpy arrays reduces DataLoader overhead from 34% to 8% of step time
3. **BERT over XGBoost+TF-IDF:** Contextual embeddings capture word order and semantics, critical for distinguishing overlapping-vocabulary categories (+3.2% accuracy)
4. **BERT over BiLSTM+GloVe:** Bidirectional attention + large-scale pre-training eliminates need for domain-specific embeddings

## Project Structure

```
nlp-news-classifier/
├── src/
│   ├── model/
│   │   ├── __init__.py
│   │   └── bert_classifier.py    # BERT model + label smoothing loss
│   ├── __init__.py
│   ├── preprocessing.py           # Data loading, cleaning, tokenization, caching
│   ├── train.py                   # Training script with MLflow tracking
│   └── serve.py                   # FastAPI inference server
├── tests/
│   ├── __init__.py
│   └── test_preprocessing.py      # Unit tests for preprocessing pipeline
├── configs/
│   └── default.yaml               # Default training configuration
├── Dockerfile                     # Multi-stage production build
├── requirements.txt
├── .gitignore
└── README.md
```

## Quick Start

### Training

```bash
# Install dependencies
pip install -r requirements.txt

# Train with default settings
python src/train.py \
    --model bert-base-uncased \
    --lr 2e-5 \
    --epochs 5 \
    --batch_size 32 \
    --dropout 0.3 \
    --max_len 256 \
    --output_dir checkpoints/

# Train with multi-layer attention head (better boundary performance)
python src/train.py \
    --model bert-base-uncased \
    --lr 2e-5 \
    --epochs 5 \
    --batch_size 32 \
    --dropout 0.3 \
    --max_len 256 \
    --multi_layer_head \
    --output_dir checkpoints/
```

### Inference

```bash
# Start API server
uvicorn src.serve:app --host 0.0.0.0 --port 8000

# Classify an article
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple announced a new AI chip for machine learning workloads"}'
```

### Docker

```bash
# Build
docker build -t news-classifier .

# Run
docker run -p 8000:8000 -v ./checkpoints:/app/checkpoints news-classifier
```

### Testing

```bash
pytest tests/ -v --cov=src
```

## Environment

- Python 3.10
- PyTorch 2.0.1, CUDA 11.8
- Transformers 4.30.0
- Training hardware: NVIDIA RTX 3090 (24GB VRAM), 64GB RAM

## Dataset

**AG News** (via HuggingFace Datasets, Apache 2.0 license)
- 120,000 training samples, 7,600 test samples
- 4 classes: World, Sports, Business, Sci/Tech
- Split: 80/10/10 stratified by label
- Cleaning: Near-duplicate removal by hashing (title + first 100 chars), ~2.3% removed

## Reproducibility

Seeds set in: `random`, `numpy`, `torch`, `torch.cuda`, `transformers`. CUDNN deterministic mode enabled. Remaining nondeterminism: CUDA atomicAdd in multi-head attention backward pass.

## Experiment Tracking

MLflow (local). Key experiment comparison:

| Run | LR | Dropout | Val F1 | Decision |
|-----|----|---------|--------|----------|
| `run_bert_lr2e5_drop03_ep5` | 2e-5 | 0.3 | **0.9412** | Selected — stable convergence |
| `run_bert_lr5e5_drop03_ep5` | 5e-5 | 0.3 | 0.9187 | Rejected — catastrophic forgetting |
| `run_bert_lr2e5_drop05_ep5` | 2e-5 | 0.5 | 0.9301 | Rejected — underfitting |
| `run_bert_lr3e5_drop03_ep5` | 3e-5 | 0.3 | 0.9378 | Close second, slightly less stable |

## License

MIT — compatible with all upstream dependencies (Apache 2.0, BSD-3).
