# from catchMinor.base.base_torch_model import TorchBaseModel
import math

import torch
import torch.nn as nn

from catchMinor.time_series_model.AnomalyTransformer.at_config import (
    AnomalyTransformer_config,
)


class AnomalyAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.softmax = nn.Softmax(dim=2)

    def forward(
        self,
        Sigma: torch.Tensor,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None,
        dk: int = 64,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # |Sigma| = (batch_size, input_length, 1)
        # |Q|     = (batch_size, input_length, dk)
        # |K|     = (batch_size, input_length, dk)
        # |V|     = (batch_size, input_length, dk)
        # |mask|  = (batch_size, input_length, input_length)

        # get Prior-Assocication
        batch_size, input_length = Sigma.size()[0], Sigma.size()[1]
        dist = torch.zeros((input_length, input_length))
        for i in range(input_length):
            for j in range(input_length):
                dist[i][j] = abs(i - j)
        dist = dist.unsqueeze(0).repeat(batch_size, 1, 1)
        kernel_weight = (
            1
            / (math.sqrt(2 * math.pi) * Sigma)
            * torch.exp(-(dist**2) / (2 * Sigma**2))
        )
        row_sum = torch.sum(kernel_weight, dim=2)
        prior_association = kernel_weight / row_sum
        # |prior-prior_association| = (batch_size, input_length, input_length)

        # get Series-Association``
        w = torch.bmm(Q, K.transpose(1, 2))
        # |w| = (batch_size, input_length, input_length)
        if mask is not None:
            assert (
                w.size() == mask.size()
            ), "weight size and mask size are differenct :("
            w.masked_fill_(mask, -float("inf"))

        series_association = self.softmax(w / (dk**0.5))
        # |series_association| = (batch_size, input_length, input_length)
        out = torch.bmm(series_association, V)
        # |out| = (batch_size, input_length, dk)

        return prior_association, series_association, out


class MultiHead(nn.Module):
    def __init__(self, hidden_size: int = 512, n_splits: int = 8):
        super().__init__()

        self.hidden_size = hidden_size
        self.n_splits = n_splits

        # we don't have to declare each linear layer, separately
        self.Sigma_linear = nn.Linear(n_splits, n_splits, bias=False)
        self.Q_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.K_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)

        self.attn = AnomalyAttention()

    def forward(self, Sigma, Q, K, V, mask=None):
        # |Sigma| = (batch_size, input_length, n_splits)
        # |Q|     = (batch_size, input_length, hidden_size)
        # |K|     = (batch_size, input_length, hidden_size)
        # |V|     = (batch_size, input_length, hidden_size) = |K|
        # |mask|  = (batch_size, input_length, input_length)

        # split it for multihead attention
        SWs = self.Sigma_linear(Sigma).split(self.n_splits // self.n_splits, dim=-1)
        QWs = self.Q_linear(Q).split(self.hidden_size // self.n_splits, dim=-1)
        KWs = self.K_linear(K).split(self.hidden_size // self.n_splits, dim=-1)
        VWs = self.V_linear(V).split(self.hidden_size // self.n_splits, dim=-1)
        # |SW_i| = (batch_size, input_length, n_splits / n_splits)
        # |QW_i| = (batch_size, input_length, hidden_size / n_splits)
        # |KW_i| = (batch_size, input_length, hidden_size / n_splits)
        # |VW_i| = (batch_size, input_length, hidden_size / n_splits) = |KW_i|

        SWs = torch.cat(SWs, dim=0)
        QWs = torch.cat(QWs, dim=0)
        KWs = torch.cat(KWs, dim=0)
        VWs = torch.cat(VWs, dim=0)
        # |SWs| = (batch_size * n_splits, input_length, n_splits / n_splits)
        # |QWs| = (batch_size * n_splits, input_length, hidden_size / n_splits)
        # |KWs| = (batch_size * n_splits, input_length, hidden_size / n_splits)
        # |VWs| = (batch_size * n_splits, input_length, hidden_size / n_splits) = |KWs|

        if mask is not None:
            mask = torch.cat([mask for _ in range(self.n_splits)], dim=0)
            # |mask| = (batch_size * n_splits, input_length, input_length)

        prior_association, series_association, out = self.attn(
            SWs,
            QWs,
            KWs,
            VWs,
            mask=mask,
            dk=self.hidden_size // self.n_splits,
        )
        # |prior_association| = (batch_size * n_splits, input_length, input_length)
        # |series_association| = (batch_size * n_splits, input_length, input_length)
        # |out| = (batch_size * n_splits, input_length, hidden_size / n_splits)

        out = out.split(Q.size(0), dim=0)
        # |c_i| = (batch_size, input_length, hidden_size / n_splits)
        out = self.linear(torch.cat(out, dim=-1))
        # |c| = (batch_size, input_length, hidden_size)

        return prior_association, series_association, out


class EncoderBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int = 512,
        n_splits: int = 8,
        dropout_p: float = 0.1,
        use_leaky_relu: bool = False,
    ):
        super().__init__()

        self.attn = MultiHead(hidden_size, n_splits)
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn_dropout = nn.Dropout(dropout_p)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.LeakyReLU() if use_leaky_relu else nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.fc_norm = nn.LayerNorm(hidden_size)
        self.fc_dropout = nn.Dropout(dropout_p)

    def forward(self, x, mask):
        prior_association, series_association, out = self.attn(x, x, x, mask)
        z = self.attn_norm(self.attn_dropout(out) + x)
        encoder_out = self.fc_norm(self.fc_dropout(self.fc(z)) + z)
        # |z| = (batch_size, n, hidden_size)

        return prior_association, series_association, encoder_out, mask


class AnomalyTransformer(nn.Module):
    """AnomalyTransformer: https://arxiv.org/pdf/2110.02642.pdf"""

    def __init__(self, config: AnomalyTransformer_config):
        self.input_length = config.input_length
        self.feature_dim = config.feature_dim
        self.hidden_size = config.hidden_size
        self.n_splits = config.n_splits
        self.n_enc_blocks = config.n_enc_blocks
        self.dropout_p = config.dropout_p

        super().__init__()

        self.emb_enc = nn.Embedding(config.input_length, config.hidden_size)
        self.emb_dropout = nn.Dropout(config.dropout_p)

        self.pos_enc = self._generate_pos_enc(config.hidden_size, config.input_length)

        self.encoder = nn.ModuleList(
            [
                EncoderBlock(
                    config.hidden_size,
                    config.n_splits,
                    config.dropout_p,
                    config.use_leaky_relu,
                )
                for _ in range(config.n_enc_blocks)
            ],
        )

        self.projection = nn.Linear(config.hidden_size, config.feature_dim, bias=True)

    @torch.no_grad()
    def _generate_pos_enc(self, hidden_size: int, max_length: int):
        enc = torch.FloatTensor(max_length, hidden_size).zero_()
        # |enc| = (max_length, hidden_size)

        pos = torch.arange(0, max_length).unsqueeze(-1).float()
        dim = torch.arange(0, hidden_size // 2).unsqueeze(0).float()
        # |pos| = (max_length, 1)
        # |dim| = (1, hidden_size // 2)

        enc[:, 0::2] = torch.sin(pos / 1e4 ** dim.div(float(hidden_size)))
        enc[:, 1::2] = torch.cos(pos / 1e4 ** dim.div(float(hidden_size)))

        return enc

    def _position_encoding(self, x, init_pos=0):
        # |x| = (batch_size, input_length, hidden_size)
        # |self.pos_enc| = (max_length, hidden_size)
        assert x.size(-1) == self.pos_enc.size(-1)
        assert x.size(1) + init_pos <= self.max_length

        pos_enc = self.pos_enc[init_pos : init_pos + x.size(1)].unsqueeze(0)
        # |pos_enc| = (1, input_length, hidden_size)
        x = x + pos_enc.to(x.device)

        return x

    def forward(self, x, y):
        # x = y, for reconstruction
        # |x| = (batch_size, input_length)
        # |y| = (batch_size, input_length)

        x = self.emb_dropout(self._position_encoding(self.emb_enc(x)))
        prior_association_list = []
        series_association_list = []
        for enc in self.encoder:
            prior_association, series_association, x, _ = enc(x)
            prior_association_list.append(prior_association)
            series_association_list.append(series_association)
        y_hat = self.projection(x)
        # |y_hat| = (batch_size, input_length, feature_dim)
        return prior_association_list, series_association_list, y_hat


# NOTE: 참고
# - https://github.com/kh-kim/simple-nmt/blob/master/simple_nmt/models/transformer.py
# - https://github.com/thuml/Anomaly-Transformer/blob/main/model/AnomalyTransformer.py
