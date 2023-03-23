import torch
import torch.nn as nn
import torch.nn.functional as F

from meloetta.frameworks.porygon.model import config
from meloetta.data import WEATHERS, PSEUDOWEATHERS


class WeatherEncoder(nn.Module):
    def __init__(self, gen: int, config: config.WeatherEncoderConfig):
        super().__init__()

        self.weather_onehot = nn.Embedding.from_pretrained(torch.eye(len(WEATHERS) + 1))
        self.time_left_onehot = nn.Embedding.from_pretrained(torch.eye(10))
        self.min_time_left_onehot = nn.Embedding.from_pretrained(torch.eye(7))

        pw_min_onehot = nn.Embedding.from_pretrained(torch.eye(8))
        self.pw_min_onehot = nn.Sequential(pw_min_onehot, nn.Flatten(2))
        pw_max_onehot = nn.Embedding.from_pretrained(torch.eye(10))
        self.pw_max_onehot = nn.Sequential(pw_max_onehot, nn.Flatten(2))

        lin_in = (
            self.weather_onehot.embedding_dim
            + self.time_left_onehot.embedding_dim
            + self.min_time_left_onehot.embedding_dim
            + pw_min_onehot.embedding_dim * len(PSEUDOWEATHERS)
            + pw_max_onehot.embedding_dim * len(PSEUDOWEATHERS)
        )
        self.lin = nn.Sequential(
            nn.Linear(lin_in, config.embedding_dim),
            nn.ReLU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )

    def forward(
        self,
        weather: torch.Tensor,
        pseudoweather: torch.Tensor,
    ):
        weather_token = weather[..., 0]
        time_left = weather[..., 1]
        min_time_left = weather[..., 2]

        weather_onehot = self.weather_onehot(weather_token + 1)
        time_left_onehot = self.time_left_onehot(time_left)
        min_time_left_onehot = self.min_time_left_onehot(min_time_left)

        pseudo_weather_x = pseudoweather + 1
        pw_min_time_left = pseudo_weather_x[..., 0]
        pw_max_time_left = pseudo_weather_x[..., 1]

        pw_min_time_left_onehot = self.pw_min_onehot(pw_min_time_left)
        pw_max_time_left_onehot = self.pw_max_onehot(pw_max_time_left)

        weather_raw = torch.cat(
            (
                weather_onehot,
                time_left_onehot,
                min_time_left_onehot,
                pw_max_time_left_onehot,
                pw_min_time_left_onehot,
            ),
            dim=-1,
        )
        weather_emb = self.lin(weather_raw)
        return weather_emb
