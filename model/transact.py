import torch
import torch.nn as nn

"""
This is a trial to re-implement the model architecture in paper

    TransAct: Transformer-based Realtime User Action Model for Recommendation at Pinterest
    https://arxiv.org/abs/2306.00248

with certain simplification

    1. the time window masking is removed for now since we would use MovieLen dataset initially

Code reference:
    1. https://github.com/pinterest/transformer_user_action/tree/main
"""

class TransActModule(nn.Module):
    def __init__(self):
        super().__init__()

        # TODO: move to model config
        self.max_seq_len = 10  # the number of sequence length to use
        self.d_action = 16
        self.d_item = 128
        self.top_k = 3

        # this is the embedding for action type in the sequences
        self.action_embedding = nn.Embedding(num_embeddings=5,
                                             embedding_dim=self.d_action, 
                                             padding_idx=0)
        # for now we would reuse the existing encoder layer available in pytorch for simplicity
        # we would use our customized encoder implementation
        self.encoding_layer = nn.TransformerEncoderLayer(
            d_model=self.d_action + 2 * self.d_item,
            nhead=1,
            dim_feedforward=1024,
            batch_first=True,  # this makes sure it is B x seq x dim
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoding_layer, num_layers=2)

    def forward(self, action_sequence, item_sequence, candidate):
        """
        This is a simplified version, where we would ignore the request time in the original
        paper which is used for time window masking

            action_sequence: the action type in user history behavior, B x seq
            item_sequence: the item embedding in user history behavior, B x seq x d_item
            candidate: the candidate item embedding to be scored, B x d_item
        """

        action_seq = action_sequence[:, :self.max_seq_len]  # B x seq
        action_seq_embedding = self.action_embedding(action_seq + 1)  # B x seq x d_action
        item_seq_embedding = item_sequence[:, :self.max_seq_len, :]  # B x seq x d_item

        # for know we would assume that there is no padding in the input seq (which could be controlled outside)
        padding_mask = action_seq <= 0  # B x seq
        attn_mask = torch.zeros_like(padding_mask, dtype=torch.float)
        attn_mask.masked_fill_(padding_mask, float("-inf"))

        action_item_embedding = torch.concat((action_seq_embedding, item_seq_embedding), dim=2)  # B x seq x (d_action + d_item)

        # the paper also mentioned that they adopted concat candidate embedding to each action_pin embedding instead
        # of append it to the sequence (which would result in seq + 1)
        candidate_embedding = candidate.unsqueeze(1).expand(-1, self.max_seq_len, -1)  # B x seq x d_item
        action_item_embedding = torch.concat((action_item_embedding, candidate_embedding), dim=-1)  # B x seq x (d_action + 2 * d_item)

        trans_out = self.encoder(src=action_item_embedding,
                                 src_key_padding_mask=attn_mask)  # B x seq x d_model

        # max pooling
        max_pool_embedding = trans_out.max(dim=1)  # B x d_model
        out = torch.concat((trans_out[:, :self.top_k, :].flatten(1), max_pool_embedding), dim=1)  # B x (top_k + 1) * d_model
        return out


        

