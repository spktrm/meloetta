import json

from dataclasses import dataclass

from typing import Any, Sequence

from meloetta.frameworks.mewzero.model import config
from meloetta.frameworks.mewzero import utils


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False


@dataclass
class AdamConfig:
    """Adam optimizer related params."""

    b1: float = 0.0
    b2: float = 0.999
    eps: float = 10e-8


@dataclass
class NerdConfig:
    """Nerd related params."""

    beta: float = 2.0
    clip: float = 10_000


@dataclass
class MewZeroConfig:
    """Learning pararms"""

    # The batch size to use when learning/improving parameters.
    batch_size: int = 32
    # The learning rate for `params`.
    learning_rate: float = 5e-5
    # The config related to the ADAM optimizer used for updating `params`.
    adam: AdamConfig = AdamConfig()
    # All gradients values are clipped to [-clip_gradient, clip_gradient].
    clip_gradient: float = 10_000
    # The "speed" at which `params_target` is following `params`.
    target_network_avg: float = 0.01

    # RNaD algorithm configuration.
    # Entropy schedule configuration. See EntropySchedule class documentation.
    entropy_schedule_repeats: Sequence[int] = (1,)
    entropy_schedule_size: Sequence[int] = (50,)
    # The weight of the reward regularisation term in RNaD.
    eta_reward_transform: float = 0.2
    gamma: float = 1.0
    nerd: NerdConfig = NerdConfig()
    c_vtrace: float = 1.0

    trajectory_length: int = 512

    # battle_format = "gen8randomdoublesbattle"
    # battle_format = "gen8randombattle"
    battle_format: str = "gen9randombattle"
    # battle_format = "gen8ou"
    # battle_format = "gen9ou"
    # battle_format = "gen8doublesou"
    # battle_format: str = "gen3randombattle"
    # battle_format = "gen9doublesou"

    gen, format = utils.get_gen_and_gametype(battle_format)

    team: str = "null"
    # team = "charizard||heavydutyboots|blaze|furyswipes,scaleshot,toxic,roost||85,,85,85,85,85||,0,,,,||88|"
    # team = "charizard||heavydutyboots|blaze|hurricane,fireblast,toxic,roost||85,,85,85,85,85||,0,,,,||88|]venusaur||blacksludge|chlorophyll|leechseed,substitute,sleeppowder,sludgebomb||85,,85,85,85,85||,0,,,,||82|]blastoise||whiteherb|torrent|shellsmash,earthquake,icebeam,hydropump||85,85,85,85,85,85||||86|"
    # team = "charizard||heavydutyboots|blaze|hurricane,fireblast,toxic,roost||85,,85,85,85,85||,0,,,,||88|]blastoise||whiteherb|torrent|shellsmash,earthquake,icebeam,hydropump||85,85,85,85,85,85||||86|"
    # team = "ceruledge||lifeorb|weakarmor|bitterblade,closecombat,shadowsneak,swordsdance||85,85,85,85,85,85||||82|,,,,,fighting]grafaiai||leftovers|prankster|encore,gunkshot,knockoff,partingshot||85,85,85,85,85,85||||86|,,,,,dark]greedent||sitrusberry|cheekpouch|bodyslam,psychicfangs,swordsdance,firefang||85,85,85,85,85,85||||88|,,,,,psychic]quaquaval||lifeorb|moxie|aquastep,closecombat,swordsdance,icespinner||85,85,85,85,85,85||||80|,,,,,fighting]flapple||lifeorb|hustle|gravapple,outrage,dragondance,suckerpunch||85,85,85,85,85,85||||84|,,,,,grass]pachirisu||assaultvest|voltabsorb|nuzzle,superfang,thunderbolt,uturn||85,85,85,85,85,85||||94|,,,,,flying"

    actor_device: str = "cpu"
    learner_device: str = "cuda"

    debug_mode = False

    # This config will spawn 20 workers with 2 players each
    # for a total of 40 players, playing 20 games.
    # it is recommended to have an even number of players per worker
    num_actors: int = 1 if debug_mode else 12
    num_buffers: int = max(4 * num_actors, 2 * batch_size)

    model_config: config.MewZeroModelConfig = config.MewZeroModelConfig()

    eval: bool = not debug_mode

    def __getindex__(self, key: str):
        return self.__dict__[key]

    def __repr__(self):
        d = {
            key: value if is_jsonable(value) else repr(value)
            for key, value in self.__dict__.items()
        }
        body = json.dumps(d, indent=4, sort_keys=True)
        return f"MewZeroConfig({body})"

    def get(self, key: Any, default=None):
        return self.__dict__.get(key, default)
