from typing import NamedTuple

from meloetta.data import CHOICE_FLAGS


class _Config(NamedTuple):
    @classmethod
    def from_dict(self, d: dict):
        return self(**d)


class SideEncoderConfig(_Config):
    entity_embedding_dim: int = 256
    num_layers: int = 3
    num_heads: int = 2
    key_size: int = entity_embedding_dim // 2
    value_size: int = key_size
    resblocks_num_before: int = 2
    resblocks_num_after: int = 2
    resblocks_hidden_size: int = key_size

    output_dim: int = 512


class ScalarEncoderConfig(_Config):
    embedding_dim: int = SideEncoderConfig.output_dim
    num_resblocks: int = 2


class EncoderConfig(_Config):
    side_encoder_config: SideEncoderConfig = SideEncoderConfig()
    scalar_encoder_config: ScalarEncoderConfig = ScalarEncoderConfig()


class CoreConfig(_Config):
    side_encoder_dim: int = SideEncoderConfig.output_dim
    scalar_encoder_dim: int = ScalarEncoderConfig.embedding_dim

    raw_embedding_dim: int = SideEncoderConfig.output_dim
    hidden_dim: int = raw_embedding_dim
    num_layers: int = 2


class ValueHeadConfig(_Config):
    hidden_dim: int = CoreConfig.hidden_dim
    num_resblocks: int = 2


class ActionTypeHeadConfig(_Config):
    input_dim: int = CoreConfig.hidden_dim
    num_layers_resnet: int = 2
    num_layers_mlp: int = 2
    num_actions: int = 3


class FlagHeadConfig(_Config):
    input_dim: int = CoreConfig.hidden_dim
    num_layers_resnet: int = 1
    num_layers_mlp: int = 1
    num_actions: int = len(CHOICE_FLAGS)


class MoveHeadConfig(_Config):
    input_dim: int = CoreConfig.hidden_dim
    num_layers_resnet: int = 2

    query_dim: int = input_dim
    num_layers_query: int = 2

    key_dim: int = SideEncoderConfig.entity_embedding_dim
    num_layers_key: int = 2


class SwitchHeadConfig(_Config):
    input_dim: int = CoreConfig.hidden_dim
    num_layers_resnet: int = 2

    query_dim: int = input_dim
    num_layers_query: int = 2

    key_dim: int = SideEncoderConfig.entity_embedding_dim
    num_layers_key: int = 2


class PolicyHeadsConfig(_Config):
    action_type_head_config: ActionTypeHeadConfig = ActionTypeHeadConfig()
    flag_head_config: FlagHeadConfig = FlagHeadConfig()
    move_head_config: MoveHeadConfig = MoveHeadConfig()
    switch_head_config: SwitchHeadConfig = SwitchHeadConfig()


def _scale(config: _Config, factor: float):
    datum = {}
    for key in config.__annotations__.keys():
        value = getattr(config, key)
        if hasattr(value, "__annotations__"):
            datum[key] = _scale(value, factor)
        elif isinstance(value, int) or isinstance(value, float):
            datum[key] = max(int(value * factor), 1)

    return config.from_dict(datum)


class ProximaModelConfig(_Config):
    encoder_config: EncoderConfig = EncoderConfig()
    policy_heads_config: PolicyHeadsConfig = PolicyHeadsConfig()
    resnet_core_config: CoreConfig = CoreConfig()
    value_head_config: ValueHeadConfig = ValueHeadConfig()

    @classmethod
    def from_scale(self, scale: float) -> "ProximaModelConfig":
        return _scale(self(), scale)


def main():
    config = ProximaModelConfig.from_scale(0.2)
    print(config.encoder_config.side_encoder_config.entity_embedding_dim)


if __name__ == "__main__":
    main()
