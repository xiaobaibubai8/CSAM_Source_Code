import torch
import torch.nn as nn
import torch.nn.functional as F
import fasttext
import numpy as np
from torch_geometric.nn import GCNConv, GlobalAttention, TransformerConv
from pooling1d import pooling1d
from torch_geometric.utils import add_self_loops

class GCNfusion(nn.Module):

    def __init__(self, pool_type='mean', code_vocab_size=10000, desc_vocab_size=10000,
                 embedding_size=128, hidden_dim=256, device='cuda:0', node_aggr='global_attn'):
        super(GCNfusion, self).__init__()
        # code word embedding
        self.code_word_embedding = nn.Embedding(code_vocab_size, embedding_size)
        self.code_word_embedding2 = nn.Embedding(10000, embedding_size)
        self.code_pad_id = 0
        # desc word embedding
        self.desc_word_embedding = nn.Embedding(desc_vocab_size, embedding_size)
        self.desc_pad_id = 0
        self.hidden_dim = hidden_dim
        
        self.conv2 = GCNConv(embedding_size, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, embedding_size)
        
        self.conv1 = TransformerConv(in_channels=128, out_channels=16, heads=8).to(device)
        self.node_aggr = node_aggr
        if self.node_aggr == 'global_attn':
            self.node_global_attention = GlobalAttention(nn.Linear(embedding_size, 1))
        self.pooling = pooling1d(pool_type)
        self.device = device

    def forward(self, batch_graphs, batch_desc, batch_lens, return_embedding=False,
                cross_entropy=False,
                normalize_sim_matrix=True):
        h_n = self.encode_desc(batch_desc, batch_lens)
        mean_repr = self.encode_code_graph(batch_graphs)
        if not return_embedding:
            if not cross_entropy:
                pos_scores = F.cosine_similarity(mean_repr, h_n)
                return pos_scores
            else:
                if normalize_sim_matrix:
                    mean_repr = F.normalize(mean_repr, dim=-1, p=2)
                    h_n = F.normalize(h_n, dim=-1, p=2)
                return torch.mm(mean_repr, h_n.T)
        else:
            return mean_repr, h_n

    def encode_desc(self, batched_descriptions, batched_descriptions_lens):
        desc_embedding = self.desc_word_embedding(batched_descriptions)
        max_seq_len = batched_descriptions.size()[1]
        tokens_mask = (batched_descriptions.ne(self.desc_pad_id)).to(self.device)
        batch_desc_lens = torch.sum(batched_descriptions != self.desc_pad_id, dim=-1).cpu()
        desc_hidden2 = self.pooling(input_emb=desc_embedding, input_mask=tokens_mask,
                                    input_len=batch_desc_lens.to(self.device))
        
        desc_hidden = self.pooling(input_emb=desc_embedding, input_mask=tokens_mask,
                                    input_len=batch_desc_lens.to(self.device))
        return desc_hidden

    # graph feature
    def encode_code_graph(self, batch_graphs):
        # [batch_size, max_code_len]
        outer_stmt_features = batch_graphs.x                            
        edge_index = batch_graphs.edge_index                            
        mini_x = batch_graphs.mini_x                                    
        mini_edge_index = batch_graphs.mini_edge                        
        mini_x_batch = batch_graphs.mini_x_batch                        
        mini_edge_batch = batch_graphs.mini_edge_batch                  
        
        #【CFG_num, embedding_size】
        stmt_embedding = self.code_word_embedding2(outer_stmt_features)
       
        tokens_mask = (outer_stmt_features.ne(self.code_pad_id)).to(self.device)
        batch_code_lens = torch.sum(outer_stmt_features != self.code_pad_id, dim=-1).to(self.device)
        stmt_embedding = self.pooling(input_emb=stmt_embedding, input_mask=tokens_mask, input_len=batch_code_lens)
        
        #【ast_num, embedding_size】
        mini_stmt_embedding = self.code_word_embedding(mini_x)
        mini_tokens_mask = (mini_x.ne(self.code_pad_id)).to(self.device)
        mini_batch_code_lens = torch.sum(mini_x != self.code_pad_id, dim=-1).to(self.device)
        mini_stmt_embedding = self.pooling(input_emb=mini_stmt_embedding, input_mask=mini_tokens_mask, input_len=mini_batch_code_lens)
        
        #【ast_num, embedding_size】
        mini_stmt_hidden, (mini_edge_index, attention_weights) = self.conv1(mini_stmt_embedding, mini_edge_index, return_attention_weights=True)
        
        # 【CFG Node, embedding_size】
        mini_function_repr = self.node_global_attention(mini_stmt_hidden, mini_x_batch)
       
        # fuse statement text feature and structure feature
        mini_function_repr = (mini_function_repr + stmt_embedding) / 2
        
        # set two layers
        final_stmt = self.conv2(mini_function_repr, edge_index)
        # relu for GCN
        final_stmt = F.relu(final_stmt)
        final_stmt = self.conv3(final_stmt, edge_index)
        
        if self.node_aggr == 'global_attn':
            function_repr = self.node_global_attention(final_stmt, batch_graphs.batch)
        else:
            function_repr = self.q(final_stmt, batch_graphs.batch, edge_index = edge_index)
        # [batch_size, embedding_size]
        return function_repr
