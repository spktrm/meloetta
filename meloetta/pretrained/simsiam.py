import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random

from tqdm import tqdm


from meloetta.frameworks.nash_ketchum.model.config import EncoderConfig
from meloetta.frameworks.nash_ketchum.model.encoders import (
    PrivateEncoder,
    PublicEncoder,
    ScalarEncoder,
    WeatherEncoder,
)

from meloetta.data import (
    GENDERS,
)


class Siamese(nn.Module):
    def __init__(self, gen: int, n_active: int, config: EncoderConfig) -> None:
        super().__init__()

        self.private_encoder = PrivateEncoder(
            gen=gen, n_active=n_active, config=config.private_encoder_config
        )
        self.public_encoder = PublicEncoder(
            gen=gen, n_active=n_active, config=config.public_encoder_config
        )
        self.weather_encoder = WeatherEncoder(
            gen=gen, config=config.weather_encoder_config
        )
        self.scalar_encoder = ScalarEncoder(
            gen=gen, n_active=n_active, config=config.scalar_encoder_config
        )
        moves_dim = EncoderConfig.private_encoder_config.entity_embedding_dim
        self.moves = nn.Sequential(
            nn.Linear(moves_dim, moves_dim),
            nn.ReLU(),
            nn.Linear(moves_dim, moves_dim),
        )
        switches_dim = EncoderConfig.private_encoder_config.entity_embedding_dim
        self.switches = nn.Sequential(
            nn.Linear(switches_dim, switches_dim),
            nn.ReLU(),
            nn.Linear(switches_dim, switches_dim),
        )
        private_entity_emb_dim = (
            EncoderConfig.private_encoder_config.entity_embedding_dim
        )
        self.private_entity_emb = nn.Sequential(
            nn.Linear(private_entity_emb_dim, private_entity_emb_dim),
            nn.ReLU(),
            nn.Linear(private_entity_emb_dim, private_entity_emb_dim),
        )
        public_entity_emb_dim = EncoderConfig.public_encoder_config.entity_embedding_dim
        self.public_entity_emb = nn.Sequential(
            nn.Linear(public_entity_emb_dim, public_entity_emb_dim),
            nn.ReLU(),
            nn.Linear(public_entity_emb_dim, public_entity_emb_dim),
        )
        public_scalar_emb_dim = EncoderConfig.scalar_encoder_config.embedding_dim
        self.public_scalar_emb = nn.Sequential(
            nn.Linear(public_scalar_emb_dim, public_scalar_emb_dim),
            nn.ReLU(),
            nn.Linear(public_scalar_emb_dim, public_scalar_emb_dim),
        )
        weather_emb_dim = EncoderConfig.weather_encoder_config.embedding_dim
        self.weather_emb = nn.Sequential(
            nn.Linear(weather_emb_dim, weather_emb_dim),
            nn.ReLU(),
            nn.Linear(weather_emb_dim, weather_emb_dim),
        )
        scalar_emb_dim = EncoderConfig.scalar_encoder_config.embedding_dim
        self.scalar_emb = nn.Sequential(
            nn.Linear(scalar_emb_dim, scalar_emb_dim),
            nn.ReLU(),
            nn.Linear(scalar_emb_dim, scalar_emb_dim),
        )

    def sample_(self, batch_size: int) -> torch.Tensor:

        team_shape = (batch_size, 7)
        batch = torch.zeros(1, *team_shape, 28, dtype=torch.long)

        batch[..., 0] = torch.randint(
            0,
            self.private_encoder.ability_embedding._modules["0"].num_embeddings - 1,
            team_shape,
        )
        batch[..., 1] = torch.randint(0, 2, team_shape)
        batch[..., 2] = torch.randint(0, 2, team_shape)
        batch[..., 3] = torch.randint(0, len(GENDERS), team_shape)
        batch[..., 4] = torch.randint(0, 2047, team_shape)
        batch[..., 5] = torch.randint(
            0,
            self.private_encoder.item_embedding._modules["0"].num_embeddings - 1,
            team_shape,
        )
        batch[..., 6] = torch.randint(
            0, self.private_encoder.level_sqrt_onehot.num_embeddings - 1, team_shape
        )
        batch[..., 7] = torch.randint(0, 2047, team_shape)
        batch[..., 4] = batch[..., 4].clamp(max=batch[..., 7])
        batch[..., 8] = torch.randint(
            0,
            self.private_encoder.pokedex_embedding._modules["0"].num_embeddings - 1,
            team_shape,
        )
        batch[..., 9] = torch.randint(
            0, self.private_encoder.forme_embedding.num_embeddings - 1, team_shape
        )
        batch[..., 10] = torch.randint(
            0, self.private_encoder.atk_sqrt_onehot.num_embeddings - 1, team_shape
        )
        batch[..., 11] = torch.randint(
            0, self.private_encoder.def_sqrt_onehot.num_embeddings - 1, team_shape
        )
        batch[..., 12] = torch.randint(
            0, self.private_encoder.spa_sqrt_onehot.num_embeddings - 1, team_shape
        )
        batch[..., 13] = torch.randint(
            0, self.private_encoder.spd_sqrt_onehot.num_embeddings - 1, team_shape
        )
        batch[..., 14] = torch.randint(
            0, self.private_encoder.spe_sqrt_onehot.num_embeddings - 1, team_shape
        )
        batch[..., 15] = torch.randint(
            0, self.private_encoder.status_onehot.num_embeddings - 1, team_shape
        )
        batch[..., 16] = torch.randint(
            0, self.private_encoder.commanding_onehot.num_embeddings - 1, team_shape
        )
        batch[..., 17] = torch.randint(
            0, self.private_encoder.reviving_onehot.num_embeddings - 1, team_shape
        )
        batch[..., 18] = torch.randint(
            0, self.private_encoder.teratype_onehot.num_embeddings - 1, team_shape
        )
        batch[..., 19] = torch.randint(
            0, self.private_encoder.tera_onehot.num_embeddings - 1, team_shape
        )
        batch[..., -8::2] = torch.randint(
            0, self.private_encoder.move_embedding.num_embeddings - 1, team_shape + (4,)
        )
        batch[..., -7::2] = torch.randint(0, 63, team_shape + (4,))
        return batch.detach()

    def transform_(self, batch: torch.Tensor, chance: float) -> torch.Tensor:
        batch_size = batch.shape[1]

        team_shape = (1,)
        for bi in range(batch_size):
            for ti in range(batch.shape[2]):
                for i in range(21):
                    c = random.random()
                    if c <= chance:
                        if i == 0:
                            batch[..., bi, ti, 0] = torch.randint(
                                0,
                                self.private_encoder.ability_embedding._modules[
                                    "0"
                                ].num_embeddings
                                - 1,
                                team_shape,
                            )
                        if i == 1:
                            batch[..., bi, ti, 1] = torch.randint(0, 2, team_shape)
                        if i == 2:
                            batch[..., bi, ti, 2] = torch.randint(0, 2, team_shape)
                        if i == 2:
                            batch[..., bi, ti, 3] = torch.randint(
                                0, len(GENDERS), team_shape
                            )
                        if i == 3:
                            batch[..., bi, ti, 4] = torch.randint(0, 2047, team_shape)
                        if i == 4:
                            batch[..., bi, ti, 5] = torch.randint(
                                0,
                                self.private_encoder.item_embedding._modules[
                                    "0"
                                ].num_embeddings
                                - 1,
                                team_shape,
                            )
                        if i == 5:
                            batch[..., bi, ti, 6] = torch.randint(
                                0,
                                self.private_encoder.level_sqrt_onehot.num_embeddings
                                - 1,
                                team_shape,
                            )
                        if i == 6:
                            batch[..., bi, ti, 7] = torch.randint(0, 2047, team_shape)
                            batch[..., bi, ti, 4] = batch[..., bi, ti, 4].clamp(
                                max=batch[..., bi, ti, 7]
                            )
                        if i == 7:
                            batch[..., bi, ti, 8] = torch.randint(
                                0,
                                self.private_encoder.pokedex_embedding._modules[
                                    "0"
                                ].num_embeddings
                                - 1,
                                team_shape,
                            )
                        if i == 8:
                            batch[..., bi, ti, 9] = torch.randint(
                                0,
                                self.private_encoder.forme_embedding.num_embeddings - 1,
                                team_shape,
                            )
                        if i == 10:
                            batch[..., bi, ti, 10] = torch.randint(
                                0,
                                self.private_encoder.atk_sqrt_onehot.num_embeddings - 1,
                                team_shape,
                            )
                        if i == 11:
                            batch[..., bi, ti, 11] = torch.randint(
                                0,
                                self.private_encoder.def_sqrt_onehot.num_embeddings - 1,
                                team_shape,
                            )
                        if i == 12:
                            batch[..., bi, ti, 12] = torch.randint(
                                0,
                                self.private_encoder.spa_sqrt_onehot.num_embeddings - 1,
                                team_shape,
                            )
                        if i == 13:
                            batch[..., bi, ti, 13] = torch.randint(
                                0,
                                self.private_encoder.spd_sqrt_onehot.num_embeddings - 1,
                                team_shape,
                            )
                        if i == 14:
                            batch[..., bi, ti, 14] = torch.randint(
                                0,
                                self.private_encoder.spe_sqrt_onehot.num_embeddings - 1,
                                team_shape,
                            )
                        if i == 15:
                            batch[..., bi, ti, 15] = torch.randint(
                                0,
                                self.private_encoder.status_onehot.num_embeddings - 1,
                                team_shape,
                            )
                        if i == 16:
                            batch[..., bi, ti, 16] = torch.randint(
                                0,
                                self.private_encoder.commanding_onehot.num_embeddings
                                - 1,
                                team_shape,
                            )
                        if i == 17:
                            batch[..., bi, ti, 17] = torch.randint(
                                0,
                                self.private_encoder.reviving_onehot.num_embeddings - 1,
                                team_shape,
                            )
                        if i == 18:
                            batch[..., bi, ti, 18] = torch.randint(
                                0,
                                self.private_encoder.teratype_onehot.num_embeddings - 1,
                                team_shape,
                            )
                        if i == 19:
                            batch[..., bi, ti, 19] = torch.randint(
                                0,
                                self.private_encoder.tera_onehot.num_embeddings - 1,
                                team_shape,
                            )
                        if i == 20:
                            batch[..., bi, ti, -8::2] = torch.randint(
                                0,
                                self.private_encoder.move_embedding.num_embeddings - 1,
                                team_shape + (4,),
                            )
                        if i == 21:
                            batch[..., bi, ti, -7::2] = torch.randint(
                                0, 63, team_shape + (4,)
                            )
        return batch.detach()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z11, z12, z13 = self.encoder(x1)  # NxC
        z21, z22, z23 = self.encoder(x2)  # NxC

        p11 = self.predictor1(z11)  # NxC
        p12 = self.predictor2(z12)  # NxC
        p13 = self.predictor3(z13)  # NxC

        p21 = self.predictor1(z21)  # NxC
        p22 = self.predictor2(z22)  # NxC
        p23 = self.predictor3(z23)  # NxC

        return (
            p11,
            p12,
            p13,
            z11,
            z12,
            z13,
            p21,
            p22,
            p23,
            z21,
            z22,
            z23,
        )


def loss_fn(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    z = z.detach()

    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return -(p * z).sum(dim=1).mean()


def main():
    batch_size = 16
    chance = 0.5
    device = "cuda"

    model = Siamese()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = loss_fn

    progress = tqdm(range(1000))
    for _ in progress:
        with torch.no_grad():
            x = model.sample_(batch_size)
            x1 = model.transform_(x.clone(), chance)
            x2 = model.transform_(x.clone(), chance)

        x1 = x1.to(device)
        x2 = x2.to(device)

        (p11, p12, p13, z11, z12, z13, p21, p22, p23, z21, z22, z23) = model(x1, x2)
        p11 = torch.flatten(p11, 0, -2)
        p12 = torch.flatten(p12, 0, -2)
        p13 = torch.flatten(p13, 0, -2)
        z11 = torch.flatten(z11, 0, -2)
        z12 = torch.flatten(z12, 0, -2)
        z13 = torch.flatten(z13, 0, -2)
        p21 = torch.flatten(p21, 0, -2)
        p22 = torch.flatten(p22, 0, -2)
        p23 = torch.flatten(p23, 0, -2)
        z21 = torch.flatten(z21, 0, -2)
        z22 = torch.flatten(z22, 0, -2)
        z23 = torch.flatten(z23, 0, -2)

        loss1 = criterion(p11, z21) + criterion(p21, z11)
        loss2 = criterion(p12, z22) + criterion(p22, z12)
        loss3 = criterion(p13, z23) + criterion(p23, z13)

        loss = (loss1 + loss2 + loss3) / 6

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress.set_description(
            f"l1: {loss1.item():.3f}, l2: {loss2.item():.3f}, l3: {loss3.item():.3f}"
        )

    torch.save(
        model.encoder.state_dict(),
        "meloetta/frameworks/nash_ketchum/model/encoders/private_encoder.pt",
    )


if __name__ == "__main__":
    main()
