import torch.nn as nn

from meloetta.frameworks.porygon.model.interfaces import (
    State,
    EncoderOutput,
    SideEncoderOutput,
)
from meloetta.frameworks.porygon.model.config import EncoderConfig
from meloetta.frameworks.porygon.model.encoders import (
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
        # self.obs_decoder = ObservationDecoder(gen=gen, config=config)

    def forward(self, state: State) -> EncoderOutput:
        sides = state["sides"]
        boosts = state["boosts"]
        volatiles = state["volatiles"]
        side_conditions = state["side_conditions"]
        pseudoweathers = state["pseudoweathers"]
        weather = state["weather"]
        wisher = state["wisher"]
        turn = state["turn"]
        turns_since_last_move = state["turns_since_last_move"]
        n = state["n"]
        total_pokemon = state["total_pokemon"]
        faint_counter = state["faint_counter"]

        # action masks
        action_type_mask = state["action_type_mask"]
        move_mask = state["move_mask"]
        switch_mask = state["switch_mask"]
        flag_mask = state["flag_mask"]
        max_move_mask = state.get("max_move_mask")
        target_mask = state.get("target_mask")

        side_encoder_output = self.side_encoder.forward(
            sides,
            boosts,
            volatiles,
            side_conditions,
            wisher,
        )

        weather_emb = self.weather_encoder(
            weather,
            pseudoweathers,
        )

        scalar_emb = self.scalar_encoder(
            turn,
            turns_since_last_move,
            action_type_mask,
            move_mask,
            max_move_mask,
            switch_mask,
            flag_mask,
            target_mask,
            n,
            total_pokemon,
            faint_counter,
        )

        return EncoderOutput(
            moves=side_encoder_output.moves,
            switches=side_encoder_output.switches,
            side_embedding=side_encoder_output.side_embedding,
            private_entity=side_encoder_output.private_entity,
            public_entity=side_encoder_output.public_entity,
            weather_emb=weather_emb,
            scalar_emb=scalar_emb,
        )
