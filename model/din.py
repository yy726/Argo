import torch
import torch.nn as nn
import torch.nn.functional as F

"""
This is a trial to re-implement the model architecture in paper

    Deep Interest Network for Click-Through Rate Prediction
    https://arxiv.org/pdf/1706.06978

with certain simplification
"""


class LocalActivationUnit(nn.Module):
    """
        This is the unit to compute the attention score  between the target item and the item within user
        historical behavior sequence.
    """
    def __init__(self, hidden_size=[80, 40], embedding_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(in_features=embedding_dim*4, out_features=hidden_size[0], bias=True)
        self.fc2 = nn.Linear(in_features=hidden_size[0], out_features=hidden_size[1], bias=True)
        self.fc3 = nn.Linear(in_features=hidden_size[1], out_features=1, bias=True)

    def forward(self, query, user_history):
        """
            query: candidate item, B x 1 x embedding_dim
            user_history: sequence of item user has interacted with, B x seq_len x embedding_dim

            out: attention score, B x seq_len x 1 
        """

        seq_len = user_history.size(1)
        # here we expand the original query to B * seq_len * embedding_dim for concat
        queries = torch.cat([query for _ in range(seq_len)], dim=1)  # B * seq_len * embedding_dim
        # this residual is something not mentioned in the paper but in the code, also here use element-wise product instead
        # of out product on the query and key embeddings 
        attention_input = torch.cat([queries, user_history, queries - user_history, queries * user_history], dim=-1)

        attention_out = self.fc1(attention_input)
        # TODO use PReLU/Dice as activation function in the future
        attention_out = F.relu(attention_out)
        attention_out = self.fc2(attention_out)
        attention_out = F.relu(attention_out)
        return self.fc3(attention_out)


class UserHistoryBehaviorPoolingModule(nn.Module):
    """
        This is the module to handle the interaction of candidate ad with user history behavior sequence,
        compute attention scores of each history behavior item and then compute a sum pooling as the final representation,
        this is a `dynamic` approach to compute a different history sequence representation for each candidate
    """

    def __init__(self):
        super().__init__()

        self.local_attn = LocalActivationUnit(hidden_size=[64, 32], embedding_dim=8)

    def forward(self, query, user_history, user_history_length):
        """
            query: candidate item, B x 1 x embedding_dim
            user_history: sequence of item user has interacted with, B x seq_len x embedding_dim
            user_history_length: length of user history, for masking, B x 1

            out: hidden representation of user history, B x 1 x embedding_dim
        """

        attention_scores = self.local_attn(query, user_history)  # B x seq_len x 1
        attention_scores = torch.transpose(attention_scores, 1, 2)  # B x 1 x seq_len

        # need to mask the attention scores due to various user history length and padding (similar to HF attention mask)
        # this broadcasting logic is usually hard to understand, here is a short explanation
        # for masking, our goal is to generate a mask for each sample in the batch to mask the attention score for the
        # sequence, thus it is of shape B x 1 x seq_len; the `torch.arange(user_history.size(1))[None, :]` would generate
        # a position of 1 x max_len and `user_history_length[:, None]` would expand the original `user_history_length`
        # from B x 1 to B x 1 x 1, then when the broadcast logic happens, where we expand 1 x max_len to B x 1 x max_len,
        # a.k.a each row is repeated B times; and B x 1 x 1 to B x 1 x max_len, where the actual length is repeated max_len
        # times
        mask = torch.arange(user_history.size(1))[None, :] < user_history_length[:, None]  # B x 1 x seq_len

        masked_attention_scores = attention_scores * mask  # B x 1 x seq_len

        # attention scores weighting
        out = torch.matmul(masked_attention_scores, user_history)  # B x 1 x embedding_dim
        return out


if __name__ == "__main__":
    B = 3
    max_len = 6
    embedding_dim = 8
    
    query = torch.ones((B, 1, embedding_dim))
    user_history = torch.ones((B, max_len, embedding_dim))
    user_history_length = torch.tensor([[2], [4], [6]])

    user_behavior_pooling = UserHistoryBehaviorPoolingModule()

    out = user_behavior_pooling(query, user_history, user_history_length)
    print(out)




