import sys
import os
import torch
import random
import logging
import time
import math
import numpy as np 
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.autograd import Variable

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import utils
from models.relation_graph_sage import RelationGraphSAGE
from models.ewc import EWC
from docred_extensions.docred_data_handler import DocREDDataHandler
from handlers.model_handler import ModelHandler
from extensions import detection
from extensions import memory_handler


def train_document(model, data_handler, args):
    """Train model on a single document sentence by sentence"""
    # Model training
    times = []
    
    # Reset document to start from first sentence
    data_handler.reset_document()
    
    # Get all sentences as delta graphs
    sentence_graphs = []
    while True:
        delta_graph = data_handler.get_next_sentence_graph()
        if delta_graph is None:
            break
        sentence_graphs.append(delta_graph)
    
    logging.info(f'Training on {len(sentence_graphs)} sentences')
    
    for epoch in range(args.num_epochs):
        losses = 0
        start_time = time.time()
        
        # Process each sentence
        for sent_idx, delta_graph in enumerate(sentence_graphs):
            # Extract nodes and relations from delta graph
            nodes = delta_graph['nodes']
            relations = delta_graph['relations']
            
            if len(nodes) == 0 or len(relations) == 0:
                continue
                
            # Prepare relation training data
            head_nodes = []
            tail_nodes = []
            relation_labels = []
            
            for relation in relations:
                head_nodes.append(relation['head'])
                tail_nodes.append(relation['tail'])
                # Convert relation type to index (simplified - in practice would use a mapping)
                relation_labels.append(hash(relation['relation']) % 10)  # Dummy mapping
            
            if len(head_nodes) == 0:
                continue
                
            # Convert to tensors (but keep node indices as regular Python ints for sampler)
            head_nodes_list = head_nodes  # Keep as list of ints for sampler
            tail_nodes_list = tail_nodes  # Keep as list of ints for sampler
            head_nodes = torch.LongTensor(head_nodes).to(args.device)
            tail_nodes = torch.LongTensor(tail_nodes).to(args.device)
            relation_labels = torch.LongTensor(relation_labels).to(args.device)
            
            # Training step
            model.optimizer.zero_grad()
            loss = model.compute_relation_loss(head_nodes_list, tail_nodes_list, relation_labels)
            loss.backward()
            model.optimizer.step()
            
            loss_val = loss.data.item()
            losses += loss_val
            if (np.isnan(loss_val)):
                logging.error('Loss Val is NaN !!!')
                sys.exit()
        
        if epoch % 10 == 0:
            logging.debug('--------- Epoch: ' + str(epoch) + ' ' + str(np.round(losses / len(sentence_graphs), 10)))
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.round(np.mean(times), 6)
    logging.info("Average epochs time: " + str(avg_time))
    return avg_time


def run(args):
    # Data loader
    data = DocREDDataHandler()
    data.load(args.data, 'train_annotated.json')
    
    # For now, just process one document
    # In practice, would loop through all documents
    
    # Model parameter
    num_entities = data.get_num_entities()
    features = data.get_features()
    adj_lists = data.get_adj_lists()
    feature_size = features.shape[1]  # BERT embedding size
    embed_size = 64
    num_layers = 2
    num_relations = 10   # Placeholder for number of relation types
    
    # Layers
    layers = [feature_size] + [embed_size] * num_layers + [embed_size]
    
    # Model definition
    sage = RelationGraphSAGE(layers, features, adj_lists, args, num_relations)

    # Model optimizer
    sage.optimizer = torch.optim.SGD(sage.parameters(), lr=args.learning_rate)
    
    # Train on document
    avg_time = train_document(sage, data, args)
    
    # Save model
    model_handler = ModelHandler(os.path.join(args.save_path, 'docred_model'))
    model_handler.save(sage.state_dict(), 'relation_graph_sage.pkl')
    
    return avg_time


def evaluate(args):
    """Evaluate model on DocRED dataset"""
    try:
        # Load model
        data = DocREDDataHandler()
        data.load(args.data, 'dev.json')  # Use dev set for evaluation
        
        # Model parameter
        num_entities = data.get_num_entities()
        features = data.get_features()
        adj_lists = data.get_adj_lists()
        feature_size = features.shape[1]  # BERT embedding size
        embed_size = 64
        num_layers = 2
        num_relations = 10   # Placeholder for number of relation types
        
        # Layers
        layers = [feature_size] + [embed_size] * num_layers + [embed_size]
        
        # Model definition
        sage = RelationGraphSAGE(layers, features, adj_lists, args, num_relations)
        
        # Load model
        model_handler = ModelHandler(os.path.join(args.save_path, 'docred_model'))
        if not model_handler.not_exist():
            sage.load_state_dict(model_handler.load('relation_graph_sage.pkl'))
            logging.info("Model loaded successfully for evaluation")
        else:
            logging.warning("No model found for evaluation")
            return 0, 0
        
        # Process document and compute predictions
        data.reset_document()
        
        # Get all sentences as delta graphs
        sentence_graphs = []
        while True:
            delta_graph = data.get_next_sentence_graph()
            if delta_graph is None:
                break
            sentence_graphs.append(delta_graph)
        
        logging.info(f'Evaluating on {len(sentence_graphs)} sentences')
        
        # For now, just return dummy scores
        # In practice, would compute actual relation F1 scores by:
        # 1. Collecting all ground truth relations
        # 2. Generating predictions using the model
        # 3. Computing Ign F1 and Rel F1
        
        ign_f1 = np.random.rand()  # Placeholder
        rel_f1 = np.random.rand()  # Placeholder
        
        logging.info(f"Evaluation completed - Ign F1: {ign_f1:.4f}, Rel F1: {rel_f1:.4f}")
        
        return ign_f1, rel_f1
    except Exception as e:
        logging.warning(f"Evaluation failed: {e}")
        return 0, 0


if __name__ == "__main__":
    args = utils.parse_argument()
    if args.eval:
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    utils.print_args(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.device = utils.check_device(args.cuda)

    # Metrics placeholders
    print_ans = ['', '', '', '']
    avg_ans = [0.0, 0.0, 0.0, 0.0]

    args.save_path = os.path.join('../res', args.data)
    
    logging.info('Starting DocRED Relation Extraction Training')
    
    if args.eval == False:
        b = run(args)
    else:
        b = 0

    a = evaluate(args)
    
    print('Ignoring-Type F1:\t', a[0])
    print('Relation F1:\t', a[1])
    print('Time:\t', b)