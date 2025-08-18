import os
import sys
import json
import numpy as np
import logging
import random
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import BERT
try:
    from transformers import BertTokenizer, BertModel
    import torch
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    logging.warning("BERT libraries not available. Using dummy embeddings.")

class DocREDDataHandler(object):
    def __init__(self, use_bert=True):
        super(DocREDDataHandler, self).__init__()
        self.documents = []
        self.current_doc_idx = 0
        self.current_sent_idx = 0
        self.use_bert = use_bert and BERT_AVAILABLE
        
        import os
import sys
import json
import numpy as np
import logging
import random
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import BERT
try:
    from transformers import BertTokenizer, BertModel
    import torch
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    logging.warning("BERT libraries not available. Using dummy embeddings.")

class DocREDDataHandler(object):
    def __init__(self, use_bert=True):
        super(DocREDDataHandler, self).__init__()
        self.documents = []
        self.current_doc_idx = 0
        self.current_sent_idx = 0
        self.use_bert = use_bert and BERT_AVAILABLE
        
        if self.use_bert:
            try:
                # Use local BERT model
                bert_model_path = "/root/.cache/modelscope/hub/models/google-bert/bert-base-uncased"
                if os.path.exists(bert_model_path):
                    logging.info(f"Loading BERT model from local path: {bert_model_path}")
                    self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)
                    self.bert_model = BertModel.from_pretrained(bert_model_path)
                else:
                    # Fallback to downloading
                    logging.info("Loading BERT model from remote repository")
                    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                    self.bert_model = BertModel.from_pretrained('bert-base-uncased')
                    
                self.bert_model.eval()  # Set to evaluation mode
                logging.info("BERT model initialized for entity embeddings")
            except Exception as e:
                logging.warning(f"Failed to initialize BERT model: {e}. Using dummy embeddings.")
                self.use_bert = False
        else:
            logging.info("Using dummy embeddings (BERT not enabled or not available)")

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
                
        # Create features for entities
        if self.use_bert:
            self.features = self._extract_bert_entity_embeddings()
        else:
            # Create dummy features for entities (768-dim for BERT)
            self.features = np.random.rand(len(self.entities), 768)
            
        logging.info(f"Entity features shape: {self.features.shape}")
        
    def _extract_bert_entity_embeddings(self):
        """Extract BERT embeddings for each entity"""
        entity_embeddings = []
        
        # Get the full document text
        full_document = []
        for sent in self.sentences:
            full_document.extend(sent)
        document_text = " ".join(full_document)
        
        # Process each entity
        for entity_idx, entity_mentions in enumerate(self.entities):
            mention_embeddings = []
            
            # Process each mention of the entity
            for mention in entity_mentions:
                sent_id = mention['sent_id']
                pos = mention['pos']  # [start, end] positions
                sent_tokens = self.sentences[sent_id]
                
                # Get the mention text
                mention_text = " ".join(sent_tokens[pos[0]:pos[1]])
                
                # Tokenize and get BERT embeddings
                if len(mention_text.strip()) > 0:
                    try:
                        inputs = self.tokenizer(mention_text, return_tensors="pt", truncation=True, max_length=512, padding=True)
                        with torch.no_grad():
                            outputs = self.bert_model(**inputs)
                            # Use the CLS token embedding or average of token embeddings
                            mention_embedding = outputs.last_hidden_state[:, 0, :].numpy()  # CLS token
                            mention_embeddings.append(mention_embedding.flatten())
                    except Exception as e:
                        logging.warning(f"Error processing entity {entity_idx} mention: {e}")
                        # Fallback to average of tokens
                        try:
                            inputs = self.tokenizer(mention_text, return_tensors="pt", truncation=True, max_length=512, padding=True)
                            with torch.no_grad():
                                outputs = self.bert_model(**inputs)
                                # Average of all token embeddings
                                mention_embedding = torch.mean(outputs.last_hidden_state[:, 1:, :], dim=1).numpy()
                                mention_embeddings.append(mention_embedding.flatten())
                        except Exception as e2:
                            logging.warning(f"Fallback failed for entity {entity_idx} mention: {e2}")
            
            # Aggregate embeddings from all mentions
            if mention_embeddings:
                # Average all mention embeddings
                entity_embedding = np.mean(mention_embeddings, axis=0)
            else:
                # Fallback to dummy embedding if all mentions failed
                logging.warning(f"All mentions failed for entity {entity_idx}, using dummy embedding")
                entity_embedding = np.random.rand(768)
                
            entity_embeddings.append(entity_embedding)
            
        return np.array(entity_embeddings)
    
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
                
        # Create features for entities
        if self.use_bert:
            self.features = self._extract_bert_entity_embeddings()
        else:
            # Create dummy features for entities (768-dim for BERT)
            self.features = np.random.rand(len(self.entities), 768)
            
        logging.info(f"Entity features shape: {self.features.shape}")
        
    def _extract_bert_entity_embeddings(self):
        """Extract BERT embeddings for each entity"""
        entity_embeddings = []
        
        # Get the full document text
        full_document = []
        for sent in self.sentences:
            full_document.extend(sent)
        document_text = " ".join(full_document)
        
        # Process each entity
        for entity_idx, entity_mentions in enumerate(self.entities):
            mention_embeddings = []
            
            # Process each mention of the entity
            for mention in entity_mentions:
                sent_id = mention['sent_id']
                pos = mention['pos']  # [start, end] positions
                sent_tokens = self.sentences[sent_id]
                
                # Get the mention text
                mention_text = " ".join(sent_tokens[pos[0]:pos[1]])
                
                # Tokenize and get BERT embeddings
                if len(mention_text.strip()) > 0:
                    try:
                        inputs = self.tokenizer(mention_text, return_tensors="pt", truncation=True, max_length=512)
                        with torch.no_grad():
                            outputs = self.bert_model(**inputs)
                            # Use the CLS token embedding or average of token embeddings
                            mention_embedding = outputs.last_hidden_state[:, 0, :].numpy()  # CLS token
                            mention_embeddings.append(mention_embedding.flatten())
                    except Exception as e:
                        logging.warning(f"Error processing entity {entity_idx} mention: {e}")
                        # Fallback to average of tokens
                        try:
                            inputs = self.tokenizer(mention_text, return_tensors="pt", truncation=True, max_length=512)
                            with torch.no_grad():
                                outputs = self.bert_model(**inputs)
                                # Average of all token embeddings
                                mention_embedding = torch.mean(outputs.last_hidden_state[:, 1:, :], dim=1).numpy()
                                mention_embeddings.append(mention_embedding.flatten())
                        except Exception as e2:
                            logging.warning(f"Fallback failed for entity {entity_idx} mention: {e2}")
            
            # Aggregate embeddings from all mentions
            if mention_embeddings:
                # Average all mention embeddings
                entity_embedding = np.mean(mention_embeddings, axis=0)
            else:
                # Fallback to dummy embedding if all mentions failed
                logging.warning(f"All mentions failed for entity {entity_idx}, using dummy embedding")
                entity_embedding = np.random.rand(768)
                
            entity_embeddings.append(entity_embedding)
            
        return np.array(entity_embeddings)
    
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