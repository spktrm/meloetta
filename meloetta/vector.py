import warnings

try:
    import torch
except ModuleNotFoundError:
    warnings.warn("torch not found")

try:
    import numpy as np
except ModuleNotFoundError:
    warnings.warn("numpy not found")

from meloetta.room import BattleRoom

from typing import Union, NamedTuple, Tuple, List, Dict, Any

from meloetta.data import (
    BOOSTS,
    VOLATILES,
    PSEUDOWEATHERS,
    SIDE_CONDITIONS,
    get_item_effect_token,
    get_status_token,
    get_gender_token,
    get_species_token,
    get_ability_token,
    get_item_token,
    get_move_token,
    get_type_token,
    get_weather_token,
    get_pseudoweather_token,
    get_side_condition_token,
)


_DEFAULT_BACKEND = "torch"
_ACTIVE_SIZE = 156
_RESERVE_SIZE = 33


class NamedVector(NamedTuple):
    vector: Union[np.ndarray, torch.Tensor]
    schema: Dict[str, Tuple[int, int]]


def to_id(string: str):
    return "".join(c for c in string if c.isalnum()).lower()


class Side(NamedTuple):
    n: torch.Tensor
    total_pokemon: torch.Tensor
    faint_counter: torch.Tensor
    side_conditions: torch.Tensor
    wisher: torch.Tensor
    active: torch.Tensor
    reserve: torch.Tensor
    stealthrock: torch.Tensor
    spikes: torch.Tensor
    toxicspikes: torch.Tensor
    stickyweb: torch.Tensor


class ReservePokemon(NamedTuple):
    species: int
    forme: int
    slot: int
    hp: int
    fainted: int
    level: int
    gender: int
    moves: List[str]
    ability: int
    base_ability: int
    item: int
    item_effect: int
    prev_item: int
    prev_item_effect: int
    terastallized: int
    status: int
    status_stage: int
    last_move: int
    times_attacked: int

    def vector(
        self,
        backend: str = _DEFAULT_BACKEND,
        with_schema: bool = False,
        *args,
        **kwargs
    ):
        arr = []
        schema = {}

        for field in sorted(self._fields):
            value = getattr(self, field)

            start = len(arr)
            if isinstance(value, int):
                arr.append(value)
            elif isinstance(value, list):
                arr += value
            finish = len(arr)
            schema[field] = (start, finish)

        if backend == "numpy":
            arr = np.array(arr, *args, **kwargs)
        elif backend == "torch":
            arr = torch.tensor(arr, *args, **kwargs)
        else:
            raise ValueError("Invalid Backend, must be one of `numpy` or `torch`")

        if with_schema:
            return NamedVector(arr, schema)
        else:
            return arr


class ActivePokemon(NamedTuple):
    species: int
    forme: int
    slot: int
    hp: int
    fainted: int
    level: int
    gender: int
    moves: List[str]
    ability: int
    base_ability: int
    item: int
    item_effect: int
    prev_item: int
    prev_item_effect: int
    terastallized: int
    status: int
    status_stage: int
    last_move: int
    times_attacked: int
    boosts: Dict[str, Any]
    volatiles: Dict[str, Any]
    sleep_turns: int
    toxic_turns: int

    def vector(
        self,
        backend: str = _DEFAULT_BACKEND,
        with_schema: bool = False,
        *args,
        **kwargs
    ):
        arr = []
        schema = {}

        for field in sorted(self._fields):
            value = getattr(self, field)

            start = len(arr)
            if isinstance(value, int):
                arr.append(value)
            elif isinstance(value, list):
                arr += value
            finish = len(arr)
            schema[field] = (start, finish)

        if backend == "numpy":
            arr = np.array(arr, *args, **kwargs)
        elif backend == "torch":
            arr = torch.tensor(arr, *args, **kwargs)
        else:
            raise ValueError("Invalid Backend, must be one of `numpy` or `torch`")

        if with_schema:
            return NamedVector(arr, schema)
        else:
            return arr


class State(NamedTuple):
    p1: Side
    p2: Side
    weather: torch.Tensor
    weather_time_left: torch.Tensor
    weather_min_time_left: torch.Tensor
    pseudo_weather: torch.Tensor
    turn: torch.Tensor
    log: List[str]


Battle = Dict[str, Any]


LAST_MOVES = set()


class VectorizedState:
    def __init__(self, room: BattleRoom, battle: Battle):
        self.room = room
        self.battle = battle
        self.gen = self.battle["dex"]["gen"]

    @classmethod
    def from_battle(self, room: BattleRoom, battle: Battle):
        vstate = VectorizedState(room, battle)
        return vstate.vectorize()

    def vectorize(self):
        p1, p2 = self._vectorize_sides()
        pseudoweathers = {
            (pseudoweather[0]).replace(" ", "").lower(): pseudoweather[1:]
            for pseudoweather in self.battle["pseudoWeather"]
        }
        pseudoweathers = torch.stack(
            [
                torch.tensor(pseudoweathers.get(pseudoweather, [-1, -1]))
                for pseudoweather in PSEUDOWEATHERS
            ]
        )
        LAST_MOVES.add(self.battle["lastMove"])
        return State(
            p1=p1,
            p2=p2,
            weather=torch.tensor(get_weather_token(self.battle["weather"])),
            weather_time_left=torch.tensor(self.battle["weatherTimeLeft"]),
            weather_min_time_left=torch.tensor(self.battle["weatherMinTimeLeft"]),
            pseudo_weather=pseudoweathers,
            turn=torch.tensor(self.battle["turn"]),
            log=self.battle["stepQueue"],
        )

    def _vectorize_sides(self):
        p1 = self._vectorize_side("mySide")
        p2 = self._vectorize_side("farSide")
        return p1, p2

    def _vectorize_side(self, side_id: str):
        side = self.battle[side_id]
        controlling = self.battle["pokemonControlled"]

        p_control = side["battle"]["pokemonControlled"]
        active_size = p_control - len(side["pokemon"][:controlling])
        active_padding = [torch.zeros(_ACTIVE_SIZE) for _ in range(active_size)]

        active = torch.stack(
            [
                self._vectorize_public_active_pokemon(p)
                for p in side["pokemon"][:controlling]
            ]
            + active_padding
        )
        active = active.view(1, 1, *active.shape)

        reserve_size = 6 - len(side["pokemon"][controlling:]) - p_control
        reserve_padding = [torch.zeros(_RESERVE_SIZE) for _ in range(reserve_size)]

        reserve = torch.stack(
            [
                self._vectorize_public_reserve_pokemon(p)
                for p in side["pokemon"][controlling:]
            ]
            + reserve_padding
        )
        reserve = reserve.view(1, 1, *reserve.shape)

        side_conditions = torch.stack(
            [
                torch.tensor(
                    side["sideConditions"].get(side_condition, [None, -1, -1, -1])[1:]
                )
                for side_condition in SIDE_CONDITIONS
                if side_condition
                not in {"stealthrock", "spikes", "toxicspikes", "stickyweb"}
            ]
        )
        side_conditions = side_conditions.view(1, 1, *side_conditions.shape)

        stealthrock = torch.tensor(
            side["sideConditions"].get("stealthrock", [None, 0])[1]
        )
        stealthrock = stealthrock.view(1, 1, *stealthrock.shape)
        spikes = torch.tensor(side["sideConditions"].get("spikes", [None, 0])[1])
        spikes = spikes.view(1, 1, *spikes.shape)
        toxicspikes = torch.tensor(
            side["sideConditions"].get("toxicspikes", [None, 0])[1]
        )
        toxicspikes = toxicspikes.view(1, 1, *toxicspikes.shape)
        stickyweb = torch.tensor(side["sideConditions"].get("stickyweb", [None, 0])[1])
        stickyweb = stickyweb.view(1, 1, *stickyweb.shape)

        return Side(
            n=torch.tensor(side["n"]),
            total_pokemon=torch.tensor(side["totalPokemon"]),
            faint_counter=torch.tensor(side["faintCounter"]),
            active=active,
            reserve=reserve,
            side_conditions=side_conditions,
            stealthrock=stealthrock,
            spikes=spikes,
            toxicspikes=toxicspikes,
            stickyweb=stickyweb,
            wisher=side["wisher"],
        )

    def _vectorize_public_active_pokemon(self, pokemon):
        if pokemon is None:
            return None

        moves = []
        for move, pp in pokemon["moveTrack"]:
            moves += [get_move_token(self.gen, "name", move), pp]
        for _ in range(8 - int(len(moves) / 2)):
            moves += [-1, -1]
        boosts = [pokemon["boosts"].get(boost, 0) for boost in BOOSTS]
        volatiles = [
            1 if volatile in pokemon["volatiles"] else 0 for volatile in VOLATILES
        ]
        forme = pokemon["speciesForme"].replace(pokemon["name"] + "-", "")
        return ActivePokemon(
            species=get_species_token(self.gen, "name", pokemon["name"]),
            forme=get_species_token(self.gen, "forme", forme),
            slot=pokemon["slot"],
            hp=pokemon["hp"] / pokemon["maxhp"],
            fainted=1 if pokemon["fainted"] else 0,
            level=pokemon["level"],
            gender=get_gender_token(pokemon["gender"]),
            moves=moves,
            ability=get_ability_token(self.gen, "name", pokemon["ability"]),
            base_ability=get_ability_token(self.gen, "name", pokemon["baseAbility"]),
            item=get_item_token(self.gen, "name", pokemon["item"]),
            item_effect=get_item_effect_token(pokemon["itemEffect"]),
            prev_item=get_item_token(self.gen, "name", pokemon["prevItem"]),
            prev_item_effect=get_item_effect_token(pokemon["prevItemEffect"]),
            terastallized=get_type_token(self.gen, pokemon["terastallized"]),
            status=get_status_token(pokemon["status"]),
            status_stage=pokemon["statusStage"],
            last_move=get_move_token(self.gen, "id", pokemon["lastMove"]),
            times_attacked=pokemon["timesAttacked"],
            boosts=boosts,
            volatiles=volatiles,
            sleep_turns=pokemon["statusData"]["sleepTurns"],
            toxic_turns=pokemon["statusData"]["toxicTurns"],
        ).vector()

    def _vectorize_public_reserve_pokemon(self, pokemon):
        moves = []
        for move, pp in pokemon["moveTrack"]:
            moves += [get_move_token(self.gen, "name", move), pp]
        for _ in range(8 - int(len(moves) / 2)):
            moves += [-1, -1]
        forme = pokemon["speciesForme"].replace(pokemon["name"] + "-", "")
        return ReservePokemon(
            species=get_species_token(self.gen, "name", pokemon["name"]),
            forme=get_species_token(self.gen, "forme", forme),
            slot=pokemon["slot"],
            hp=pokemon["hp"] / pokemon["maxhp"],
            fainted=1 if pokemon["fainted"] else 0,
            level=pokemon["level"],
            gender=get_gender_token(pokemon["gender"]),
            moves=moves,
            ability=get_ability_token(self.gen, "name", pokemon["ability"]),
            base_ability=get_ability_token(self.gen, "name", pokemon["baseAbility"]),
            item=get_item_token(self.gen, "name", pokemon["item"]),
            item_effect=get_item_effect_token(pokemon["itemEffect"]),
            prev_item=get_item_token(self.gen, "name", pokemon["prevItem"]),
            prev_item_effect=get_item_effect_token(pokemon["prevItemEffect"]),
            terastallized=get_type_token(self.gen, pokemon["terastallized"]),
            status=get_status_token(pokemon["status"]),
            status_stage=pokemon["statusStage"],
            last_move=get_move_token(self.gen, "name", pokemon["lastMove"]),
            times_attacked=pokemon["timesAttacked"],
        ).vector()


class VectorizedChoice:
    def __init__(self, room: BattleRoom, battle: Battle):
        self.room = room
        self.battle = battle
        self.gen = self.battle["dex"]["gen"]

    @classmethod
    def from_battle(self, room: BattleRoom, battle: Battle):
        vstate = VectorizedChoice(room, battle)
        return vstate.vectorize()

    def vectorize(self):
        return
