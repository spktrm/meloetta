from typing import NamedTuple


class _Config(NamedTuple):
    @classmethod
    def from_dict(self, d: dict):
        return self(**d)


class PrivateEncoderConfig(_Config):
    embedding_dim: int = 128
    entity_embedding_dim: int = 128
    transformer_nhead: int = 2
    transformer_dim_feedforward: int = 256
    transformer_num_layers: int = 3


class PublicEncoderConfig(_Config):
    entity_embedding_dim: int = 128
    scalar_embedding_dim: int = 128
    transformer_nhead: int = 2
    transformer_dim_feedforward: int = 256
    transformer_num_layers: int = 3


class ScalarEncoderConfig(_Config):
    embedding_dim: int = 128


class WeatherEncoderConfig(_Config):
    embedding_dim: int = 128


class EncoderConfig(_Config):
    private_encoder_config: PrivateEncoderConfig = PrivateEncoderConfig()
    public_encoder_config: PublicEncoderConfig = PublicEncoderConfig()
    scalar_encoder_config: ScalarEncoderConfig = ScalarEncoderConfig()
    weather_encoder_config: WeatherEncoderConfig = WeatherEncoderConfig()
    output_dim: int = (
        private_encoder_config.entity_embedding_dim
        + 2 * public_encoder_config.entity_embedding_dim
        + public_encoder_config.scalar_embedding_dim
        + weather_encoder_config.embedding_dim
        + scalar_encoder_config.embedding_dim
    )


class ResNetCoreConfig(_Config):
    raw_state_embedding_dim: int = EncoderConfig.output_dim
    projected_state_embedding_dim: int = 384
    num_resblocks: int = 2


class ValueHeadConfig(_Config):
    state_embedding_dim: int = ResNetCoreConfig.projected_state_embedding_dim
    num_resblocks: int = 2


class ActionTypeHeadConfig(_Config):
    state_embedding_dim: int = ResNetCoreConfig.projected_state_embedding_dim
    num_action_types: int = 3
    context_dim: int = ScalarEncoderConfig.embedding_dim
    residual_dim: int = 128
    action_map_dim: int = 128
    autoregressive_embedding_dim: int = 128


class FlagHeadConfig(_Config):
    autoregressive_embedding_dim: int = (
        ActionTypeHeadConfig.autoregressive_embedding_dim
    )


class MaxMoveHeadConfig(_Config):
    entity_embedding_dim: int = PrivateEncoderConfig.entity_embedding_dim
    key_dim: int = 32

    state_embedding_dim: int = ActionTypeHeadConfig.autoregressive_embedding_dim


class MoveHeadConfig(_Config):
    entity_embedding_dim: int = PrivateEncoderConfig.entity_embedding_dim
    key_dim: int = 32

    state_embedding_dim: int = ActionTypeHeadConfig.autoregressive_embedding_dim


class SwitchHeadConfig(_Config):
    entity_embedding_dim: int = PrivateEncoderConfig.entity_embedding_dim
    key_dim: int = 32

    state_embedding_dim: int = ActionTypeHeadConfig.autoregressive_embedding_dim


class TargetHeadConfig(_Config):
    pass


class PolicyHeadsConfig(_Config):
    action_type_head_config: ActionTypeHeadConfig = ActionTypeHeadConfig()
    flag_head_config: FlagHeadConfig = FlagHeadConfig()
    max_move_head_config: MaxMoveHeadConfig = MaxMoveHeadConfig()
    move_head_config: MoveHeadConfig = MoveHeadConfig()
    switch_head_config: SwitchHeadConfig = SwitchHeadConfig()
    target_head_config: TargetHeadConfig = TargetHeadConfig()


def _scale(config: _Config, factor: float):
    datum = {}
    for key, value in config._asdict().items():
        if isinstance(value, tuple) and hasattr(value, "_asdict"):
            datum[key] = _scale(value, factor)
        elif isinstance(value, int) or isinstance(value, float):
            datum[key] = max(int(value * factor), 1)

    return config.from_dict(datum)


class NAshKetchumModelConfig(_Config):
    encoder_config: EncoderConfig = EncoderConfig()
    policy_heads_config: PolicyHeadsConfig = PolicyHeadsConfig()
    resnet_core_config: ResNetCoreConfig = ResNetCoreConfig()
    value_head_config: ValueHeadConfig = ValueHeadConfig()

    @classmethod
    def from_scale(self, scale: float):
        return _scale(self(), scale)


def main():
    config = NAshKetchumModelConfig.from_scale(0.2)
    print(config)


if __name__ == "__main__":
    main()
