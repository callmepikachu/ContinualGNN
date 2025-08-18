#!/bin/bash

# For Cora dataset (original)
# python main_sage.py --data cora --num_epochs 100 --learning_rate 0.1

# For streaming Cora dataset (original)
# python main_stream.py --data cora --num_epochs 100 --learning_rate 0.1 --memory_size 100 --ewc_lambda 0.1

# For DocRED dataset (new)
# Without BERT embeddings
python main_docred.py --data docred --num_epochs 100 --learning_rate 0.01

# With BERT embeddings
# python main_docred.py --data docred --num_epochs 100 --learning_rate 0.01 --use_bert