from typing import NamedTuple


class _Config(NamedTuple):
    @classmethod
    def from_dict(self, d: dict):
        return self(**d)


class SideEncoderConfig(_Config):
    model_size: int = 128
    num_layers: int = 3
    num_heads: int = 2
    key_size: int = model_size
    value_size: int = model_size // 2
    resblocks_num_before: int = 1
    resblocks_num_after: int = 1
    resblocks_hidden_size: int = key_size
    use_layer_norm: bool = True

    entity_embedding_dim: int = model_size
    output_dim: int = 512


class ScalarEncoderConfig(_Config):
    embedding_dim: int = SideEncoderConfig.output_dim


class WeatherEncoderConfig(_Config):
    embedding_dim: int = SideEncoderConfig.output_dim


class EncoderConfig(_Config):
    side_encoder_config: SideEncoderConfig = SideEncoderConfig()
    scalar_encoder_config: ScalarEncoderConfig = ScalarEncoderConfig()
    weather_encoder_config: WeatherEncoderConfig = WeatherEncoderConfig()


class CoreConfig(_Config):
    side_encoder_dim: int = SideEncoderConfig.output_dim
    scalar_encoder_dim: int = ScalarEncoderConfig.embedding_dim
    weather_encoder_dim: int = WeatherEncoderConfig.embedding_dim

    raw_embedding_dim: int = SideEncoderConfig.output_dim
    hidden_dim: int = raw_embedding_dim
    num_layers: int = 2


class ValueHeadConfig(_Config):
    state_embedding_dim: int = CoreConfig.hidden_dim
    hidden_dim: int = state_embedding_dim
    num_resblocks: int = 2


class ActionTypeHeadConfig(_Config):
    state_embedding_dim: int = CoreConfig.hidden_dim
    num_action_types: int = 3
    residual_dim: int = state_embedding_dim


class FlagHeadConfig(_Config):
    autoregressive_embedding_dim: int = CoreConfig.hidden_dim
    hidden_dim: int = autoregressive_embedding_dim


class MoveHeadConfig(_Config):
    entity_embedding_dim: int = SideEncoderConfig.entity_embedding_dim
    autoregressive_embedding_dim: int = CoreConfig.hidden_dim
    key_dim: int = entity_embedding_dim // 4


class SwitchHeadConfig(_Config):
    entity_embedding_dim: int = SideEncoderConfig.entity_embedding_dim
    autoregressive_embedding_dim: int = CoreConfig.hidden_dim
    key_dim: int = entity_embedding_dim // 4


class TargetHeadConfig(_Config):
    pass


class PolicyHeadsConfig(_Config):
    action_type_head_config: ActionTypeHeadConfig = ActionTypeHeadConfig()
    flag_head_config: FlagHeadConfig = FlagHeadConfig()
    move_head_config: MoveHeadConfig = MoveHeadConfig()
    switch_head_config: SwitchHeadConfig = SwitchHeadConfig()
    target_head_config: TargetHeadConfig = TargetHeadConfig()


def _scale(config: _Config, factor: float):
    datum = {}
    for key in config.__annotations__.keys():
        value = getattr(config, key)
        if hasattr(value, "__annotations__"):
            datum[key] = _scale(value, factor)
        elif isinstance(value, int) or isinstance(value, float):
            datum[key] = max(int(value * factor), 1)

    return config.from_dict(datum)


class NAshKetchumModelConfig(_Config):
    encoder_config: EncoderConfig = EncoderConfig()
    policy_heads_config: PolicyHeadsConfig = PolicyHeadsConfig()
    resnet_core_config: CoreConfig = CoreConfig()
    value_head_config: ValueHeadConfig = ValueHeadConfig()

    @classmethod
    def from_scale(self, scale: float) -> "NAshKetchumModelConfig":
        return _scale(self(), scale)


def main():
    config = NAshKetchumModelConfig.from_scale(0.2)
    print(config.encoder_config.side_encoder_config.entity_embedding_dim)


if __name__ == "__main__":
    main()
