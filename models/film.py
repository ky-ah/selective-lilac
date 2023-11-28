import torch.nn as nn
from einops import reduce
from models.networks import GruEncoder, Resnet18Encoder, FiLM


class FiLMedModel(nn.Module):
    def __init__(self, cfg, comp_struct):
        super().__init__()

        # Vision model
        self.vision_encoder = Resnet18Encoder(cfg.arch.resnet_out_layer)

        # Language model
        self.language_model = GruEncoder(cfg)
        self.emb_dim = self.language_model.output_size

        # Final projection layer
        self.project = nn.Linear(2 ** (5 + cfg.arch.resnet_out_layer), self.emb_dim)

        # FiLM modules
        self.controllers = []
        for i in range(cfg.arch.film_modules):
            mod = FiLM(
                in_features=self.emb_dim,
                in_channels=2 ** (5 + cfg.arch.resnet_out_layer),
                config=comp_struct[i],
            )
            self.controllers.append(mod)
            self.add_module("FiLM_" + str(i), mod)

        # Summary of encoder/project, task-specific and shared submodules
        self.encoder_decoder = nn.ModuleList(
            [self.vision_encoder, self.language_model, self.project]
        )
        self.shared = nn.ModuleList([c.shared for c in self.controllers])
        self.specialized = nn.ModuleList([c.task_specific for c in self.controllers])

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
        l = self.language_model(mission)

        for controller in self.controllers:
            z1 = controller(z1, l, task)

        z1, z2, z3 = [reduce(z, "b c h w -> b c", "max") for z in [z1, z2, z3]]
        p1, z2, z3 = [self.project(z) for z in [z1, z2, z3]]

        return p1, z2, z3
