import torch
import torch.nn as nn
from einops import rearrange, reduce
from models.networks import Resnet18Encoder, TransformerEncoderLayer


class AttentionModel(nn.Module):
    def __init__(self, cfg, comp_struct):
        super().__init__()
        self.emb_dim = cfg.arch.embedding_dim

        # Vision model
        self.vision_encoder = Resnet18Encoder(cfg.arch.resnet_out_layer)
        self.encoder = nn.Linear(2 ** (5 + cfg.arch.resnet_out_layer), self.emb_dim)

        # Word embedding
        self.embedding = nn.Embedding(20, self.emb_dim, padding_idx=0)

        # Final projection layer
        self.project = nn.Linear(self.emb_dim, self.emb_dim)

        # Transformer Encoder Layers
        self.attention = []
        for i in range(cfg.arch.num_encoder_layers):
            t_layer = TransformerEncoderLayer(
                self.emb_dim, cfg.arch.nhead, cfg.arch.ffn_dim, config=comp_struct[i]
            )
            self.attention.append(t_layer)
            self.add_module("TransformerEncoderLayer_" + str(i), t_layer)

        # Summary of encoder/project, task-specific and shared submodules
        self.encoder_decoder = nn.ModuleList(
            [self.vision_encoder, self.encoder, self.embedding, self.project]
        )
        self.shared = nn.ModuleList([a.shared for a in self.attention])
        self.specialized = nn.ModuleList([a.task_specific for a in self.attention])

    def forward(self, x1, x2, x3, mission, task=None):
        """
        Input:
            x1: anchor/base image
            x2: positive implication image
            x3: negative implication image
        Output:
            p1: language-conditioned anchor image encoding
            z2: positive implication image encoding
            z3: negative implication image encoding
        """
        z1, z2, z3 = [self.vision_encoder(x) for x in [x1, x2, x3]]
        b, _, h, w = z1.shape
        z1, z2, z3 = [rearrange(z, "b c h w -> (b h w) c") for z in [z1, z2, z3]]
        z1, z2, z3 = [self.encoder(z) for z in [z1, z2, z3]]  # b*h*w x c
        z1, z2, z3 = [
            rearrange(z, "(b h w) c -> b h w c", b=b, h=h, w=w) for z in [z1, z2, z3]
        ]
        z1, z2, z3 = [
            rearrange(z, "b h w c -> b (h w) c", b=b, h=h, w=w) for z in [z1, z2, z3]
        ]

        l = self.embedding(mission)

        z1 = torch.cat((z1, l), dim=1)
        for t_layer in self.attention:
            z1 = t_layer(z1, task)

        z1, z2, z3 = [reduce(z, "b hw c -> b c", "mean") for z in [z1, z2, z3]]
        p1, z2, z3 = [self.project(z) for z in [z1, z2, z3]]

        return p1, z2, z3
