import torch
import torch.nn as nn

"""
    This is a simple trail to implement the RQ-VAE model based on the following paper:

        - https://arxiv.org/pdf/2305.05065

    Code reference:

        - https://github.com/karpathy/deep-vector-quantization/blob/main/dvq/model/quantize.py
"""


class VectorQuantization(nn.Module):
    """
        This is the vector quantization layer which is used to quantize the input vector

        Here we assume that the input vector dimension is already aligned with the internal
        codebook and thus there is no additional linear projection required.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # TODO: in the original implementations, k-means is used to initialize the codebook
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            x: B x d

            returns the quantized vector, index and the difference
        """
        
        # In karpathy's implementation, there is a kmeans initialization here,
        # plan to add it later
        
        # Some simple illustration on what this code is doing: 
        # first, it is computing the euclidean distance between the input vector and the codebook, 
        # using the formula
        #   ||x - c||^2 = ||x||^2 - 2x * c + ||c||^2, where c is the codebook, and x is the input vector
        # second, the `keepdim` is used here to ensure that the dimension is kept instead of reduced due to
        # the sum operation, so that the broadcasting can be applied correctly
        # (B, 1) - (B, N) + (1, N), due to broadcasting, the result would be (B, N) 
        dist = (
            x.pow(2).sum(dim=1, keepdim=True)
              - 2 * torch.mm(x, self.embedding.weight.T)
              + self.embedding.weight.pow(2).sum(dim=1, keepdim=True).T
        )  # B x N

        _, ind = (-dist).topk(k=1, dim=1)  # B x 1

        quantized = self.embedding(ind)  # B x d
        
        # In the google paper, the loss here is used to update the codebook, which used the 
        # stop gradient operation to construct the loss.
        # The detach operation would create a new tensor that is not part of the computation graph,
        # and thus the gradient would not flow through this tensor, which is used to simulate the
        # stop gradient effect.
        # This stop gradient is to prevent the codebook to collapse, which the quantized vector would be
        # updated while the input does not have to change to much, to prevent them moving closer to each other.
        loss = self.beta * (x - quantized.detach()).pow(2).mean() + (x.detach() - quantized).pow(2).mean()  # B,

        # This is the straight-through estimator, which is used to backpropagate the gradient since the 
        # topk is not differentiable.
        quantized = x + (quantized - x).detach()

        return quantized, ind, loss
    

class MLP(nn.Module):
    """
        This is the encoder/decoder module which is used to encode the input vector into the latent vector.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # use a simple MLP to encode
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            x: B x input_dim
        """
        return self.encoder(x)
        

class RQVAE(nn.Module):
    """
        RQ-VAE model, composed of the encoder/decoder and the vector quantization layer.

        The final loss is a combination of the reconstruction loss and the quantization loss.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_quantize: int):
        super().__init__()

        self.encoder = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        self.decoder = MLP(input_dim=output_dim, hidden_dim=hidden_dim, output_dim=input_dim)

        self.quantize_layers = nn.ModuleList(
            modules=[
                VectorQuantization(num_embeddings=128, embedding_dim=output_dim, beta=0.1)
            for _ in range(num_quantize) ]
        )

    def forward(self, x: torch.Tensor):
        """
            x: B x input_dim

            First use encoder to get the latent vector, then use the quantization layers
            to quantize the latent vector, compute residual as next input, and then use
            the decoder to get the reconstructed vector

            For simplicity, we would also return the loss in this forward pass
        """

        x = self.encoder(x)
        quantized_loss = 0
        quantized_vectors, quantized_indices = [], []
        for quantize_layer in self.quantize_layers:
            q, ind, loss = quantize_layer(x)
            quantized_loss += loss  # cumulate the quantization loss in each codebook
            quantized_vectors.append(q)  # this is used for reconstruction loss
            quantized_indices.append(ind)  # this is going to be the semantic ids
        
        # the torch stack here change n B x d tensors to B x d x n, and we sum over the last
        # dimension to get the sum of quantized vectors for each input
        x_recon = self.decoder(torch.stack(quantized_vectors, dim=2).sum(dim=-1))

        recon_loss = ((x_recon - x) ** 2).sum(dim=-1)  # B, 
        loss = (recon_loss + quantized_loss).mean()  # scalar
        
        return loss, quantized_indices

