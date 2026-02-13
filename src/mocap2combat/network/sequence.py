import torch
import torch.nn as nn
from typing import List, Tuple, Optional



class RNNNet(nn.Module):
    """
    Generic RNN-based network for RL.

    Supports:
      - GRU (default) / LSTM / vanilla RNN
      - Optional MLP head after the recurrent layer
      - Sequence input:  (B, T, input_size) if batch_first=True
        or              (T, B, input_size) if batch_first=False

    Returns:
      - output: (B, T, output_size) if return_sequences=True
                (B, output_size)   if return_sequences=False (uses last timestep)
      - next_hidden: GRU -> (num_layers, B, hidden)
                     LSTM -> ((num_layers, B, hidden), (num_layers, B, hidden))
    """
    def __init__(
        self,
        input_size: int,
        rnn_hidden_size: int,
        output_size: int,
        rnn_type: str = "gru",              # "gru" | "lstm" | "rnn"
        num_layers: int = 1,
        bidirectional: bool = False,
        batch_first: bool = True,
        rnn_dropout: float = 0.0,           # applied between stacked layers (num_layers>1)
        head_hidden_sizes: Optional[List[int]] = None,  # MLP head sizes
        head_activation: nn.Module = nn.Tanh,
        return_sequences: bool = False,
    ):
        super().__init__()

        self.rnn_type = rnn_type.lower()
        self.num_layers = num_layers
        self.rnn_hidden_size = rnn_hidden_size
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.return_sequences = return_sequences

        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM, "rnn": nn.RNN}.get(self.rnn_type)
        if rnn_cls is None:
            raise ValueError(f"Unsupported rnn_type={rnn_type}. Use 'gru', 'lstm', or 'rnn'.")

        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=rnn_hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
            dropout=rnn_dropout if num_layers > 1 else 0.0,
        )

        rnn_out_dim = rnn_hidden_size * (2 if bidirectional else 1)

        # Optional MLP "head" after the RNN
        head_hidden_sizes = head_hidden_sizes or []
        layers: List[nn.Module] = []
        prev = rnn_out_dim
        for h in head_hidden_sizes:
            layers += [nn.Linear(prev, h), head_activation()]
            prev = h
        layers += [nn.Linear(prev, output_size)]
        self.head = nn.Sequential(*layers)

    def init_hidden(self, batch_size: int, device=None):
        device = device or next(self.parameters()).device
        directions = 2 if self.bidirectional else 1

        h0 = torch.zeros(self.num_layers * directions, batch_size, self.rnn_hidden_size, device=device)
        if self.rnn_type == "lstm":
            c0 = torch.zeros(self.num_layers * directions, batch_size, self.rnn_hidden_size, device=device)
            return (h0, c0)
        return h0

    def forward(self, x: torch.Tensor, hidden=None):
        """
        x: (B,T,input) if batch_first else (T,B,input)
        hidden: optional initial hidden state
        """
        rnn_out, next_hidden = self.rnn(x, hidden)  # rnn_out: (B,T,H*) or (T,B,H*)

        if self.return_sequences:
            y = self.head(rnn_out)  # (B,T,out) or (T,B,out)
            return y, next_hidden

        # Use last timestep
        if self.batch_first:
            last = rnn_out[:, -1, :]  # (B,H*)
        else:
            last = rnn_out[-1, :, :]  # (B,H*)

        y = self.head(last)  # (B,out)
        return y, next_hidden


