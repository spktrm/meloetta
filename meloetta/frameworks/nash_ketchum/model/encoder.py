import torch.nn as nn

from meloetta.frameworks.nash_ketchum.model.interfaces import (
    State,
    EncoderOutput,
    SideEncoderOutput,
)
from meloetta.frameworks.nash_ketchum.model.config import EncoderConfig
from meloetta.frameworks.nash_ketchum.model.encoders import (
    SideEncoder,
    ScalarEncoder,
    WeatherEncoder,
)


class Encoder(nn.Module):
    def __init__(self, gen: int, n_active: int, config: EncoderConfig):
        super().__init__()
        self.side_encoder = SideEncoder(
            gen=gen, n_active=n_active, config=config.side_encoder_config
        )
        self.weather_encoder = WeatherEncoder(
            gen=gen, config=config.weather_encoder_config
        )
        self.scalar_encoder = ScalarEncoder(
            gen=gen, n_active=n_active, config=config.scalar_encoder_config
        )

    def forward(self, state: State) -> EncoderOutput:
        sides = state["sides"]
        boosts = state["boosts"]
        volatiles = state["volatiles"]
        side_conditions = state["side_conditions"]
        pseudoweathers = state["pseudoweathers"]
        weather = state["weather"]
        wisher = state["wisher"]

        # scalars
        scalars = state["scalars"]
        turn = scalars[..., 0]
        n = scalars[..., 1:3]
        total_pokemon = scalars[..., 3:5]
        faint_counter = scalars[..., 5:]

        # action masks
        action_type_mask = state["action_type_mask"]
        move_mask = state["move_mask"]
        switch_mask = state["switch_mask"]
        flag_mask = state["flag_mask"]
        max_move_mask = state.get("max_move_mask")
        target_mask = state.get("target_mask")

        side_embs = self.side_encoder(
            side=sides,
            boosts=boosts,
            volatiles=volatiles,
            side_conditions=side_conditions,
            wisher=wisher,
        )

        weather_emb = self.weather_encoder(
            weather=weather,
            pseudoweather=pseudoweathers,
        )

        scalar_emb = self.scalar_encoder(
            turn=turn,
            action_type_mask=action_type_mask,
            move_mask=move_mask,
            max_move_mask=max_move_mask,
            switch_mask=switch_mask,
            flag_mask=flag_mask,
            target_mask=target_mask,
            n=n,
            total_pokemon=total_pokemon,
            faint_counter=faint_counter,
        )

        return EncoderOutput(
            **side_embs._asdict(),
            weather_emb=weather_emb,
            scalar_emb=scalar_emb,
        )
