from typing import NamedTuple


class _Config(NamedTuple):
    @classmethod
    def from_dict(self, d: dict):
        return self(**d)


class SideEncoderConfig(_Config):
    entity_embedding_dim: int = 256

    transformer_input_size: int = entity_embedding_dim
    transformer_num_heads: int = 2
    transformer_num_layers: int = 1
    resblocks_num_before: int = 1
    resblocks_num_after: int = 1
    resblocks_hidden_size: int = 256

    output_dim: int = 256


class ScalarEncoderConfig(_Config):
    embedding_dim: int = 2 * SideEncoderConfig.output_dim


class WeatherEncoderConfig(_Config):
    embedding_dim: int = 2 * SideEncoderConfig.output_dim


class EncoderConfig(_Config):
    side_encoder_config: SideEncoderConfig = SideEncoderConfig()
    scalar_encoder_config: ScalarEncoderConfig = ScalarEncoderConfig()
    weather_encoder_config: WeatherEncoderConfig = WeatherEncoderConfig()
    output_dim: int = 2 * side_encoder_config.output_dim


class CoreConfig(_Config):
    raw_embedding_dim: int = EncoderConfig.output_dim
    hidden_dim: int = 512
    num_layers: int = 1


class ValueHeadConfig(_Config):
    state_embedding_dim: int = CoreConfig.hidden_dim
    hidden_dim: int = 256
    num_resblocks: int = 3


class FlagHeadConfig(_Config):
    autoregressive_embedding_dim: int = CoreConfig.hidden_dim
    hidden_dim: int = 256


class MaxMoveHeadConfig(_Config):
    entity_embedding_dim: int = SideEncoderConfig.entity_embedding_dim
    key_dim: int = 64

    autoregressive_embedding_dim: int = CoreConfig.hidden_dim
    query_hidden_dim: int = 256


class MoveHeadConfig(_Config):
    entity_embedding_dim: int = SideEncoderConfig.entity_embedding_dim
    key_dim: int = 64

    autoregressive_embedding_dim: int = CoreConfig.hidden_dim
    query_hidden_dim: int = 256


class SwitchHeadConfig(_Config):
    entity_embedding_dim: int = SideEncoderConfig.entity_embedding_dim
    key_dim: int = 64

    autoregressive_embedding_dim: int = CoreConfig.hidden_dim
    query_hidden_dim: int = 256


class TargetHeadConfig(_Config):
    pass


class PolicyHeadsConfig(_Config):
    flag_head_config: FlagHeadConfig = FlagHeadConfig()
    max_move_head_config: MaxMoveHeadConfig = MaxMoveHeadConfig()
    move_head_config: MoveHeadConfig = MoveHeadConfig()
    switch_head_config: SwitchHeadConfig = SwitchHeadConfig()
    target_head_config: TargetHeadConfig = TargetHeadConfig()

    entity_embedding_dim: int = SideEncoderConfig.entity_embedding_dim
    key_dim: int = MoveHeadConfig.key_dim

    autoregressive_embedding_dim: int = CoreConfig.hidden_dim
    query_hidden_dim: int = 32


def _scale(config: _Config, factor: float):
    datum = {}
    for key, value in config._asdict().items():
        if isinstance(value, tuple) and hasattr(value, "_asdict"):
            datum[key] = _scale(value, factor)
        elif isinstance(value, int) or isinstance(value, float):
            datum[key] = max(int(value * factor), 1)

    return config.from_dict(datum)


class PorygonModelConfig(_Config):
    encoder_config: EncoderConfig = EncoderConfig()
    policy_heads_config: PolicyHeadsConfig = PolicyHeadsConfig()
    resnet_core_config: CoreConfig = CoreConfig()
    value_head_config: ValueHeadConfig = ValueHeadConfig()

    @classmethod
    def from_scale(self, scale: float):
        return _scale(self(), scale)


def main():
    config = PorygonModelConfig.from_scale(0.2)
    print(config)


if __name__ == "__main__":
    main()
