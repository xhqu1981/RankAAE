import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.0, max_len=5000, batch_first=True):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        if not batch_first:
            pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe
        return self.dropout(x)

class TransformerEnergyPositionPredictor(nn.Module):
    def __init__(self, n_grid, d_model, nhead, dim_feedforward, nlayers, 
                 dropout=0.0, batch_first=True, activation='relu'):
        super(TransformerEnergyPositionPredictor, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout, 
                                              max_len=n_grid, batch_first=batch_first)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                   nhead=nhead, 
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation=activation,
                                                   batch_first=batch_first)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.input_emb = nn.Linear(1, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, 1)


    def forward(self, spec: torch.FloatTensor):
        spec = spec.unsqueeze(dim=-1)
        src = self.input_emb(spec) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.encoder(src)
        output: torch.FloatTensor = self.decoder(output)
        ene_pos = output.squeeze(dim=-1)
        return ene_pos