from collections import OrderedDict
import torch
from torch import nn
from torchvision.models import resnet18


class GruEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedding = nn.Embedding(20, cfg.arch.embedding_dim, padding_idx=0)
        self.encoder = nn.GRU(
            input_size=cfg.arch.embedding_dim,
            hidden_size=cfg.arch.gru_hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self._output_size = cfg.arch.gru_hidden_size

    def forward(self, mission):
        output, _ = self.encoder(self.embedding(mission))
        return output[:, -1]

    @property
    def output_size(self):
        return self._output_size


class Resnet18Encoder(nn.Module):
    def __init__(self, out_layer):
        super().__init__()
        layers = OrderedDict()
        for k, v in resnet18()._modules.items():
            layers[k] = v
            if k == "layer" + str(out_layer):
                break
        self.encoder = nn.Sequential(layers)

    def forward(self, x):
        return self.encoder(x)


# Inspired by FiLMedBlock from https://arxiv.org/abs/1709.07871
class FiLM(nn.Module):
    def __init__(self, in_features, in_channels, config=(0, 0, 0, 0, 0, 0)):
        super().__init__()
        self.config = config
        self.shared = nn.ModuleList()
        self.task_specific = nn.ModuleList()

        # Config 0: Weight
        if self.config[0] == 0:
            self.weight = nn.Linear(in_features, in_channels, bias=False)
            self.shared.append(self.weight)
        else:
            self.init_weight = nn.Linear(in_features, in_channels, bias=False)
            self.weight = nn.ModuleList(
                [nn.Linear(in_features, in_channels, bias=False) for _ in range(10)]
            )
            self.task_specific.append(self.weight)

        # Config 1: Bias
        if self.config[1] == 0:
            self.bias = nn.Linear(in_features, in_channels, bias=False)
            self.shared.append(self.bias)
        else:
            self.init_bias = nn.Linear(in_features, in_channels, bias=False)
            self.bias = nn.ModuleList(
                [nn.Linear(in_features, in_channels, bias=False) for _ in range(10)]
            )
            self.task_specific.append(self.bias)

        # Config 2: Convolutional layer I
        if self.config[2] == 0:
            self.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(3, 3),
                padding=1,
            )
            self.shared.append(self.conv1)
        else:
            self.init_conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(3, 3),
                padding=1,
            )
            self.conv1 = nn.ModuleList(
                [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=(3, 3),
                        padding=1,
                    )
                    for _ in range(10)
                ]
            )
            self.task_specific.append(self.conv1)

        # Config 3: Convolutional layer II
        if self.config[3] == 0:
            self.conv2 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(3, 3),
                padding=1,
            )
            self.shared.append(self.conv2)
        else:
            self.init_conv2 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(3, 3),
                padding=1,
            )
            self.conv2 = nn.ModuleList(
                [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=(3, 3),
                        padding=1,
                    )
                    for _ in range(10)
                ]
            )
            self.task_specific.append(self.conv2)

        # Config 4: Batch norm I
        if self.config[4] == 0:
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.shared.append(self.bn1)
        else:
            self.init_bn1 = nn.BatchNorm2d(in_channels)
            self.bn1 = nn.ModuleList([nn.BatchNorm2d(in_channels) for _ in range(10)])
            self.task_specific.append(self.bn1)

        # Config 5: Batch norm II
        if self.config[5] == 0:
            self.bn2 = nn.BatchNorm2d(in_channels)
            self.shared.append(self.bn2)
        else:
            self.init_bn2 = nn.BatchNorm2d(in_channels)
            self.bn2 = nn.ModuleList([nn.BatchNorm2d(in_channels) for _ in range(10)])
            self.task_specific.append(self.bn2)

    def forward(self, x, y, t):
        # Convolutional layer I
        if self.config[2] == 0:
            x = self.conv1(x)
        else:
            if t is not None:
                if isinstance(t, int):
                    x = self.conv1[t](x)
                else:
                    x_ = []
                    for i in range(len(t)):
                        x_.append(self.conv1[t[i]](x[i]))
                    x = torch.stack(x_)
            else:
                x = self.init_conv1(x)

        # Batch norm I
        if self.config[4] == 0:
            x = self.bn1(x)
        else:
            if t is not None:
                if isinstance(t, int):
                    x = self.bn1[t](x)
                else:
                    x_ = []
                    for i in range(len(t)):
                        x_.append(self.bn1[t[i]](x[i].unsqueeze(0)))
                    x = torch.cat(x_)
            else:
                x = self.init_bn1(x)

        # RelU
        x.relu_()

        # Convolutional layer II
        if self.config[3] == 0:
            x = self.conv2(x)
        else:
            if t is not None:
                if isinstance(t, int):
                    x = self.conv2[t](x)
                else:
                    x_ = []
                    for i in range(len(t)):
                        x_.append(self.conv2[t[i]](x[i]))
                    x = torch.stack(x_)
            else:
                x = self.init_conv2(x)

        # Weight
        if self.config[0] == 0:
            w = self.weight(y)
        else:
            if t is not None:
                if isinstance(t, int):
                    w = self.weight[t](y)
                else:
                    w_ = []
                    for i in range(len(t)):
                        w_.append(self.weight[t[i]](y[i]))
                    w = torch.stack(w_)
            else:
                w = self.init_weight(y)

        # Bias
        if self.config[1] == 0:
            b = self.bias(y)
        else:
            if t is not None:
                if isinstance(t, int):
                    b = self.bias[t](y)
                else:
                    b_ = []
                    for i in range(len(t)):
                        b_.append(self.bias[t[i]](y[i]))
                    b = torch.stack(b_)
            else:
                b = self.init_bias(y)

        # Linear modulation
        x = x * w.unsqueeze(-1).unsqueeze(-1) + b.unsqueeze(-1).unsqueeze(-1)

        # Batch norm II
        if self.config[5] == 0:
            x = self.bn2(x)
        else:
            if t is not None:
                if isinstance(t, int):
                    x = self.bn2[t](x)
                else:
                    x_ = []
                    for i in range(len(t)):
                        x_.append(self.bn2[t[i]](x[i].unsqueeze(0)))
                    x = torch.cat(x_)
            else:
                x = self.init_bn2(x)

        # RelU
        x.relu_()

        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout=0.1,
        batch_first=True,
        config=(0, 0, 0, 0, 0),
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.config = config
        self.shared = nn.ModuleList()
        self.task_specific = nn.ModuleList()

        # Config 0: Multihead attention
        if self.config[0] == 0:
            self.self_attn = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=batch_first
            )
            self.shared.append(self.self_attn)
        else:
            self.init_attn = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=batch_first
            )
            self.self_attn = nn.ModuleList(
                [
                    nn.MultiheadAttention(
                        d_model, nhead, dropout=dropout, batch_first=batch_first
                    )
                    for _ in range(10)
                ]
            )
            self.task_specific.append(self.self_attn)

        # Config 1: Feed-forward layer I
        if self.config[1] == 0:
            self.fc1 = nn.Linear(d_model, dim_feedforward)
            self.shared.append(self.fc1)
        else:
            self.init_fc1 = nn.Linear(d_model, dim_feedforward)
            self.fc1 = nn.ModuleList(
                [nn.Linear(d_model, dim_feedforward) for _ in range(10)]
            )
            self.task_specific.append(self.fc1)

        # Config 2: Feed-forward layer II
        if self.config[2] == 0:
            self.fc2 = nn.Linear(dim_feedforward, d_model)
            self.shared.append(self.fc2)
        else:
            self.init_fc2 = nn.Linear(dim_feedforward, d_model)
            self.fc2 = nn.ModuleList(
                [nn.Linear(dim_feedforward, d_model) for _ in range(10)]
            )
            self.task_specific.append(self.fc2)

        # Config 3: Layer normalization I
        if self.config[3] == 0:
            self.norm1 = nn.LayerNorm(d_model)
            self.shared.append(self.norm1)
        else:
            self.init_norm1 = nn.LayerNorm(d_model)
            self.norm1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(10)])
            self.task_specific.append(self.norm1)

        # Config 4: Layer normalization II
        if self.config[4] == 0:
            self.norm2 = nn.LayerNorm(d_model)
            self.shared.append(self.norm2)
        else:
            self.init_norm2 = nn.LayerNorm(d_model)
            self.norm2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(10)])
            self.task_specific.append(self.norm2)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t):
        # Layer normalization I
        if self.config[3] == 0:
            x_ = self.norm1(x)
        else:
            if t is not None:
                if isinstance(t, int):
                    x_ = self.norm1[t](x)
                else:
                    x__ = []
                    for i in range(len(t)):
                        x__.append(self.norm1[t[i]](x[i]))
                    x_ = torch.stack(x__)
            else:
                x_ = self.init_norm1(x)

        # Multihead attention
        if self.config[0] == 0:
            x_, _ = self.self_attn(x_, x_, x_)
        else:
            if t is not None:
                if isinstance(t, int):
                    x_, _ = self.self_attn[t](x_, x_, x_)
                else:
                    x__ = []
                    for i in range(len(t)):
                        x__.append(self.self_attn[t[i]](x_[i], x_[i], x_[i])[0])
                    x_ = torch.stack(x__)
            else:
                x_, _ = self.init_attn(x_, x_, x_)

        x = x + self.dropout(x_)

        # Layer normalization II
        if self.config[4] == 0:
            x_ = self.norm2(x)
        else:
            if t is not None:
                if isinstance(t, int):
                    x_ = self.norm2[t](x)
                else:
                    x__ = []
                    for i in range(len(t)):
                        x__.append(self.norm2[t[i]](x[i]))
                    x_ = torch.stack(x__)
            else:
                x_ = self.init_norm2(x)

        # Feed-forward layer I
        if self.config[1] == 0:
            x_ = self.fc1(x_)
        else:
            if t is not None:
                if isinstance(t, int):
                    x_ = self.fc1[t](x_)
                else:
                    x__ = []
                    for i in range(len(t)):
                        x__.append(self.fc1[t[i]](x_[i]))
                    x_ = torch.stack(x__)
            else:
                x_ = self.init_fc1(x_)

        # ReLU
        x_.relu_()

        # Feed-forward layer II
        if self.config[2] == 0:
            x_ = self.fc2(x_)
        else:
            if t is not None:
                if isinstance(t, int):
                    x_ = self.fc2[t](x_)
                else:
                    x__ = []
                    for i in range(len(t)):
                        x__.append(self.fc2[t[i]](x_[i]))
                    x_ = torch.stack(x__)
            else:
                x_ = self.init_fc2(x_)

        out = x + self.dropout(x_)

        return out
