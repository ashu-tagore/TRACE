import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 latent_dim: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.compress = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
        )
        self.mu      = nn.Linear(hidden_size // 2, latent_dim)
        self.log_var = nn.Linear(hidden_size // 2, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, log_var) of shape [batch, latent_dim].

        The last layer's hidden state summarizes the full sequence because
        each LSTM step has read every previous timestep.
        """
        _, (hidden, _) = self.lstm(x)
        last_hidden    = hidden[-1]
        h              = self.compress(last_hidden)
        return self.mu(h), self.log_var(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_size: int, num_layers: int,
                 output_size: int, seq_len: int, dropout: float):
        super().__init__()
        self.seq_len = seq_len
        self.expand  = nn.Sequential(
            nn.Linear(latent_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Returns reconstructed sequence of shape [batch, seq_len, n_features].

        Broadcasting z across timesteps forces the decoder to reconstruct
        a full window from a single 32-dim vector — the bottleneck makes
        it learn the most important patterns.
        """
        h           = self.expand(z)
        h_repeated  = h.unsqueeze(1).repeat(1, self.seq_len, 1)
        lstm_out, _ = self.lstm(h_repeated)
        return self.output_layer(lstm_out)


class LSTMVAE(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 latent_dim: int, seq_len: int, dropout: float):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers, latent_dim, dropout)
        self.decoder = Decoder(latent_dim, hidden_size, num_layers, input_size, seq_len, dropout)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Returns a differentiable sample z from N(mu, exp(log_var))."""
        sigma = torch.exp(0.5 * log_var)
        eps   = torch.randn_like(sigma)
        return mu + eps * sigma

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, mu, log_var). All three are needed for the VAE loss."""
        mu, log_var    = self.encoder(x)
        z              = self.reparameterize(mu, log_var)
        reconstruction = self.decoder(z)
        return reconstruction, mu, log_var