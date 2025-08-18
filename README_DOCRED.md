# ContinualGNN for DocRED

This directory contains the modified version of ContinualGNN adapted for the DocRED relation extraction task.

## Overview

The original ContinualGNN was designed for node classification in streaming graphs. This modified version adapts it for relation extraction on the DocRED dataset by:

1. Treating each document as an independent "streaming network"
2. Processing each sentence in the document as a time step
3. Extracting entity mentions as nodes and relations as edges
4. Using relation classification instead of node classification

## Key Modifications

### 1. Data Handler (`docred_extensions/docred_data_handler.py`)
- Loads DocRED JSON files
- Processes documents sentence by sentence
- Extracts entities and relations from each sentence
- Creates adjacency lists for GraphSAGE
- Supports both dummy embeddings and real BERT embeddings

### 2. Model (`models/relation_graph_sage.py`)
- Modified GraphSAGE model for relation extraction
- Replaced node classification with relation scoring
- Added bilinear scoring function for relations

### 3. Training Script (`main_docred.py`)
- Implements document-level training loop
- Sentence-by-sentence processing within each document
- Supports both training and evaluation modes

## Usage

### Training with dummy embeddings (faster)
```bash
python main_docred.py --data docred --num_epochs 100 --learning_rate 0.01
```

### Training with BERT embeddings (more accurate)
```bash
python main_docred.py --data docred --num_epochs 100 --learning_rate 0.01 --use_bert
```

### Evaluation
```bash
python main_docred.py --data docred --eval [--use_bert]
```

## Directory Structure

```
src/
├── main_docred.py              # Main training/evaluation script
├── models/
│   └── relation_graph_sage.py  # Modified GraphSAGE for relation extraction
├── docred_extensions/
│   └── docred_data_handler.py  # DocRED data processing
└── run.sh                      # Run script with all options
```

## BERT Embeddings

The system supports both dummy embeddings (for fast testing) and real BERT embeddings (for better performance). When using `--use_bert` flag:

1. The system first tries to load BERT model from local cache
2. If not found, it will attempt to download from Hugging Face
3. If both fail, it falls back to dummy embeddings

To use BERT embeddings, make sure you have:
- transformers library installed
- BERT model available (either cached locally or downloadable)

## Future Improvements

1. Implement proper relation type mapping instead of dummy hashing
2. Add memory buffer for continual learning
3. Implement EWC (Elastic Weight Consolidation) for catastrophic forgetting prevention
4. Add proper evaluation metrics (Ign F1, Rel F1) computation
5. Process multiple documents instead of just one