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

### 2. Model (`models/relation_graph_sage.py`)
- Modified GraphSAGE model for relation extraction
- Replaced node classification with relation scoring
- Added bilinear scoring function for relations

### 3. Training Script (`main_docred.py`)
- Implements document-level training loop
- Sentence-by-sentence processing within each document
- Supports both training and evaluation modes

## Usage

### Training
```bash
python main_docred.py --data docred --num_epochs 100 --learning_rate 0.01
```

### Evaluation
```bash
python main_docred.py --data docred --eval
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

## Future Improvements

1. Implement proper relation type mapping instead of dummy hashing
2. Use pre-trained BERT embeddings for entity features
3. Add memory buffer for continual learning
4. Implement EWC (Elastic Weight Consolidation) for catastrophic forgetting prevention
5. Add proper evaluation metrics (Ign F1, Rel F1) computation