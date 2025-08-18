import os
import sys
import json
import numpy as np
import logging
import random
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class DocREDDataHandler(object):
    def __init__(self):
        super(DocREDDataHandler, self).__init__()
        self.documents = []
        self.current_doc_idx = 0
        self.current_sent_idx = 0

    def load(self, data_name, data_file='train_annotated.json'):
        """Load DocRED dataset"""
        self.data_name = data_name
        data_path = os.path.join('../data', data_name, data_file)
        
        with open(data_path, 'r') as f:
            self.documents = json.load(f)
            
        logging.info(f'Loaded {len(self.documents)} documents from {data_file}')
        
        # For now, just process the first document to test
        self.current_doc = self.documents[0]
        self._process_current_document()
        
    def _process_current_document(self):
        """Process current document to extract information"""
        # Get sentences
        self.sentences = self.current_doc['sents']
        
        # Get entities (vertexSet)
        self.entities = self.current_doc['vertexSet']
        
        # Get labels (ground truth relations)
        self.labels = self.current_doc.get('labels', [])
        
        # Create entity to sentence mapping
        self.entity_to_sent = defaultdict(list)
        for entity_idx, entity_mentions in enumerate(self.entities):
            for mention in entity_mentions:
                self.entity_to_sent[entity_idx].append(mention['sent_id'])
                
        # Get title
        self.title = self.current_doc.get('title', '')
        
        logging.info(f'Processing document: {self.title}')
        logging.info(f'Number of sentences: {len(self.sentences)}')
        logging.info(f'Number of entities: {len(self.entities)}')
        logging.info(f'Number of relations: {len(self.labels)}')
        
        # Create adjacency lists for GraphSAGE (fully connected for this document)
        self.adj_lists = defaultdict(set)
        for i in range(len(self.entities)):
            for j in range(i+1, len(self.entities)):
                self.adj_lists[i].add(j)
                self.adj_lists[j].add(i)
                
        # Create dummy features for entities (768-dim for BERT)
        self.features = np.random.rand(len(self.entities), 768)
        
    def get_next_sentence_graph(self):
        """Get the next sentence as a delta graph"""
        if self.current_sent_idx >= len(self.sentences):
            return None
            
        # Create empty delta graph for this sentence
        delta_graph = {
            'nodes': [],  # Nodes (entities) in this sentence
            'node_features': [],  # Features for nodes
            'relations': [],  # Relations in this sentence (from labels)
            'sentence_id': self.current_sent_idx
        }
        
        # Find entities in this sentence
        entities_in_sent = []
        for entity_idx, entity_mentions in enumerate(self.entities):
            for mention in entity_mentions:
                if mention['sent_id'] == self.current_sent_idx:
                    entities_in_sent.append((entity_idx, mention))
                    break  # Each entity only once per sentence
                    
        # Add entities as nodes
        for entity_idx, mention in entities_in_sent:
            delta_graph['nodes'].append(entity_idx)
            # For now, using simple features - in practice, would use BERT embeddings
            delta_graph['node_features'].append({
                'name': mention['name'],
                'type': mention['type'],
                'pos': mention['pos']
            })
            
        # Find relations that involve entities in this sentence
        for label in self.labels:
            h_entity = label['h']
            t_entity = label['t']
            
            # Check if head or tail entity is in this sentence
            h_in_sent = any(mention['sent_id'] == self.current_sent_idx 
                           for mention in self.entities[h_entity])
            t_in_sent = any(mention['sent_id'] == self.current_sent_idx 
                           for mention in self.entities[t_entity])
                           
            if h_in_sent or t_in_sent:
                delta_graph['relations'].append({
                    'head': h_entity,
                    'relation': label['r'],
                    'tail': t_entity
                })
                
        self.current_sent_idx += 1
        return delta_graph
        
    def reset_document(self):
        """Reset to process the document from the beginning"""
        self.current_sent_idx = 0
        
    def get_ground_truth_relations(self):
        """Get all ground truth relations for evaluation"""
        return self.labels
        
    def get_adj_lists(self):
        """Get adjacency lists for GraphSAGE"""
        return self.adj_lists
        
    def get_features(self):
        """Get entity features"""
        return self.features
        
    def get_num_entities(self):
        """Get number of entities"""
        return len(self.entities)