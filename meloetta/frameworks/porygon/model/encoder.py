import torch.nn as nn

from meloetta.frameworks.porygon.model.interfaces import State, EncoderOutput
from meloetta.frameworks.porygon.model.config import EncoderConfig
from meloetta.frameworks.porygon.model.encoders import (
    PrivateEncoder,
    PublicEncoder,
    ScalarEncoder,
    WeatherEncoder,
    ObservationDecoder,
)


class Encoder(nn.Module):
    def __init__(self, gen: int, n_active: int, config: EncoderConfig):
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
        # self.obs_decoder = ObservationDecoder(gen=gen, config=config)

    def forward(self, state: State) -> EncoderOutput:
        # private info
        private_reserve = state["private_reserve"]

        # public info
        public_n = state["public_n"]
        public_total_pokemon = state["public_total_pokemon"]
        public_faint_counter = state["public_faint_counter"]
        public_side_conditions = state["public_side_conditions"]
        public_wisher = state["public_wisher"]
        public_active = state["public_active"]
        public_reserve = state["public_reserve"]
        public_stealthrock = state["public_stealthrock"]
        public_spikes = state["public_spikes"]
        public_toxicspikes = state["public_toxicspikes"]
        public_stickyweb = state["public_stickyweb"]

        # weather type stuff (still public)
        weather = state["weather"]
        weather_time_left = state["weather_time_left"]
        weather_min_time_left = state["weather_min_time_left"]
        pseudo_weather = state["pseudo_weather"]

        # scalar information
        turn = state["turn"]
        turns_since_last_move = state["turns_since_last_move"]
        prev_choices = state.get("prev_choices")
        choices_done = state.get("choices_done")

        # action masks
        action_type_mask = state["action_type_mask"]
        move_mask = state["move_mask"]
        switch_mask = state["switch_mask"]
        flag_mask = state["flag_mask"]
        max_move_mask = state.get("max_move_mask")
        target_mask = state.get("target_mask")

        (
            private_entity_emb,
            private_entity_raw,
            moves,
            switches,
            private_mask,
        ) = self.private_encoder(private_reserve)

        (
            public_entity_emb,
            public_scalar_emb,
            public_entities_emb,
            public_entities_raw,
            public_scalars_raw,
            public_mask,
        ) = self.public_encoder(
            public_n,
            public_total_pokemon,
            public_faint_counter,
            public_side_conditions,
            public_wisher,
            public_active,
            public_reserve,
            public_stealthrock,
            public_spikes,
            public_toxicspikes,
            public_stickyweb,
        )

        weather_emb, weather_raw = self.weather_encoder(
            weather,
            weather_time_left,
            weather_min_time_left,
            pseudo_weather,
        )

        scalar_emb, scalar_raw = self.scalar_encoder(
            turn,
            turns_since_last_move,
            prev_choices,
            choices_done,
            action_type_mask,
            move_mask,
            max_move_mask,
            switch_mask,
            flag_mask,
            target_mask,
        )

        # if self.training:
        #     decoder_ouputs = self.obs_decoder(
        #         moves,
        #         switches,
        #         public_entities_emb,
        #         public_scalar_emb,
        #         weather_emb,
        #         scalar_emb,
        #     )
        # else:
        decoder_ouputs = {}

        return EncoderOutput(
            moves=moves,
            switches=switches,
            private_entity_emb=private_entity_emb,
            public_entity_emb=public_entity_emb,
            public_scalar_emb=public_scalar_emb,
            weather_emb=weather_emb,
            scalar_emb=scalar_emb,
            # decoder targets
            private_entity_raw=private_entity_raw,
            public_entity_raw=public_entities_raw,
            public_scalars_raw=public_scalars_raw,
            weather_raw=weather_raw,
            scalar_raw=scalar_raw,
            # masks
            private_mask=private_mask,
            public_mask=public_mask,
            # decoded predictions
            **decoder_ouputs
        )
