import sys
import logging
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('..')
from layers.sage_conv import SAGEConv
from layers.sampler import Sampler
from layers.aggregator import Aggregator


class RelationGraphSAGE(nn.Module):
    def __init__(self, layers, in_features, adj_lists, args, num_relations):
        super(RelationGraphSAGE, self).__init__()

        self.layers = layers
        self.num_layers = len(layers) - 2
        self.in_features = torch.Tensor(in_features).to(args.device)
        self.adj_lists = adj_lists
        self.num_neg_samples = args.num_neg_samples
        self.device = args.device
        self.num_relations = num_relations  # Number of relation types

        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(SAGEConv(layers[i], layers[i + 1]))
        self.sampler = Sampler(adj_lists)
        self.aggregator = Aggregator()

        # Relation scorer - replaced entity classifier with relation classifier
        # Using bilinear scoring function for relations
        self.relation_weights = nn.Parameter(torch.Tensor(num_relations, layers[-2], layers[-2]))
        
        # For MLP scoring function (alternative)
        # self.relation_scorer = nn.Sequential(
        #     nn.Linear(2 * layers[-2], layers[-2]),
        #     nn.ReLU(),
        #     nn.Linear(layers[-2], num_relations)
        # )
        
        self.xent = nn.CrossEntropyLoss()

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self, nodes):
        # Convert nodes to list of ints if they are tensor
        if isinstance(nodes, torch.Tensor):
            nodes = nodes.tolist()
            
        layer_nodes, layer_mask = self._generate_layer_nodes(nodes)
        features = self.in_features[layer_nodes[0]]
        for i in range(self.num_layers):
            cur_nodes, mask = layer_nodes[i + 1], layer_mask[i]            
            aggregate_features = self.aggregator.aggregate(mask, features)
            features = self.convs[i].forward(x=features[cur_nodes], aggregate_x=aggregate_features)
        return features  # Return embeddings, not logits

    def score_relations(self, head_embeddings, tail_embeddings):
        """
        Score relations between head and tail entities
        head_embeddings: [batch_size, embed_dim]
        tail_embeddings: [batch_size, embed_dim]
        """
        batch_size = head_embeddings.size(0)
        
        # Bilinear scoring: s = h^T * W_r * t
        scores = []
        for i in range(self.num_relations):
            W_r = self.relation_weights[i]  # [embed_dim, embed_dim]
            score = torch.sum(torch.mm(head_embeddings, W_r) * tail_embeddings, dim=1)  # [batch_size]
            scores.append(score)
        
        # Stack scores: [batch_size, num_relations]
        scores = torch.stack(scores, dim=1)
        return scores
    
    def compute_relation_loss(self, head_nodes, tail_nodes, relation_labels):
        """
        Compute relation classification loss
        head_nodes: list of head node indices
        tail_nodes: list of tail node indices
        relation_labels: tensor of relation type indices
        """
        # Get embeddings for head and tail nodes
        head_embeddings = self.forward(head_nodes)  # [batch_size, embed_dim]
        tail_embeddings = self.forward(tail_nodes)  # [batch_size, embed_dim]
        
        # Score relations
        scores = self.score_relations(head_embeddings, tail_embeddings)  # [batch_size, num_relations]
        
        # Compute loss
        loss = self.xent(scores, relation_labels)
        return loss

    def loss(self, nodes, labels=None):
        # For compatibility, but in relation extraction, we use compute_relation_loss
        # This is kept for interface compatibility with EWC
        preds = self.forward(nodes)
        # Create dummy loss if needed
        dummy_loss = torch.sum(preds) * 0  # Zero loss with gradient
        return dummy_loss

    def _generate_layer_nodes(self, nodes):
        layer_nodes = list([nodes])
        layer_mask = list()
        for i in range(self.num_layers):
            nodes_idxs, unique_neighs, mask = self.sampler.sample_neighbors(layer_nodes[0])
            layer_nodes[0] = nodes_idxs
            layer_nodes.insert(0, unique_neighs)
            layer_mask.insert(0, mask.to(self.device))
        return layer_nodes, layer_mask

    def get_embeds(self, nodes):
        # Convert nodes to list of ints if they are tensor
        if isinstance(nodes, torch.Tensor):
            nodes = nodes.tolist()
            
        layer_nodes, layer_mask = self._generate_layer_nodes(nodes)
        features = self.in_features[layer_nodes[0]]
        for i in range(self.num_layers):
            cur_nodes, mask = layer_nodes[i + 1], layer_mask[i]            
            aggregate_features = self.aggregator.aggregate(mask, features)
            features = self.convs[i].forward(x=features[cur_nodes], aggregate_x=aggregate_features)
        return features  # Return embeddings