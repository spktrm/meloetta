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
from meloetta.utils import expand_bt

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
)


_DEFAULT_BACKEND = "torch"
_ACTIVE_SIZE = 156
_RESERVE_SIZE = 33


class NamedVector(NamedTuple):
    vector: Union[np.ndarray, torch.Tensor]
    schema: Dict[str, Tuple[int, int]]


class PublicSide(NamedTuple):
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


class ReservePublicPokemon(NamedTuple):
    species: torch.Tensor
    forme: torch.Tensor
    slot: torch.Tensor
    hp: torch.Tensor
    fainted: torch.Tensor
    level: torch.Tensor
    gender: torch.Tensor
    moves: torch.Tensor
    ability: torch.Tensor
    base_ability: torch.Tensor
    item: torch.Tensor
    item_effect: torch.Tensor
    prev_item: torch.Tensor
    prev_item_effect: torch.Tensor
    terastallized: torch.Tensor
    status: torch.Tensor
    status_stage: torch.Tensor
    last_move: torch.Tensor
    times_attacked: torch.Tensor

    def vector(
        self,
        backend: str = _DEFAULT_BACKEND,
        with_schema: bool = False,
        *args,
        **kwargs
    ):
        arr = []
        schema = {}

        for field in self._fields:
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


class ActivePublicPokemon(NamedTuple):
    species: torch.Tensor
    forme: torch.Tensor
    slot: torch.Tensor
    hp: torch.Tensor
    fainted: torch.Tensor
    level: torch.Tensor
    gender: torch.Tensor
    moves: torch.Tensor
    ability: torch.Tensor
    base_ability: torch.Tensor
    item: torch.Tensor
    item_effect: torch.Tensor
    prev_item: torch.Tensor
    prev_item_effect: torch.Tensor
    terastallized: torch.Tensor
    status: torch.Tensor
    status_stage: torch.Tensor
    last_move: torch.Tensor
    times_attacked: torch.Tensor
    boosts: torch.Tensor
    volatiles: torch.Tensor
    sleep_turns: torch.Tensor
    toxic_turns: torch.Tensor

    def vector(
        self,
        backend: str = _DEFAULT_BACKEND,
        with_schema: bool = False,
        *args,
        **kwargs
    ):
        arr = []
        schema = {}

        for field in self._fields:
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
    public_sides: PublicSide
    weather: torch.Tensor
    weather_time_left: torch.Tensor
    weather_min_time_left: torch.Tensor
    pseudo_weather: torch.Tensor
    turn: torch.Tensor
    log: List[str]


Battle = Dict[str, Dict[str, Dict[str, Any]]]


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
        public_sides = self._vectorize_sides()
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
        pseudoweathers = expand_bt(pseudoweathers)

        weather = torch.tensor(get_weather_token(self.battle["weather"]))
        weather = expand_bt(weather)

        weather_time_left = torch.tensor(self.battle["weatherTimeLeft"])
        weather_time_left = weather_time_left.view(
            1, 1, *(weather_time_left.shape or (1,))
        )

        weather_min_time_left = torch.tensor(self.battle["weatherMinTimeLeft"])
        weather_min_time_left = weather_min_time_left.view(
            1, 1, *(weather_min_time_left.shape or (1,))
        )

        turn = torch.tensor(self.battle["turn"])
        turn = expand_bt(turn)

        return State(
            public_sides=public_sides,
            weather=weather,
            weather_time_left=weather_time_left,
            weather_min_time_left=weather_min_time_left,
            turn=turn,
            pseudo_weather=pseudoweathers,
            log=self.battle["stepQueue"],
        )

    def _vectorize_sides(self):
        p1 = self._vectorize_public_side("mySide")
        p2 = self._vectorize_public_side("farSide")
        combined_fields = {}
        for field, value1, value2 in zip(PublicSide._fields, p1, p2):
            combined_fields[field] = torch.stack((value1, value2), dim=2)
        return PublicSide(**combined_fields)

    def _vectorize_public_side(self, side_id: str):
        side = self.battle[side_id]

        controlling = self.battle["pokemonControlled"]
        pokemon = [p for p in side["pokemon"] if not p.get("status", "") != "???"]

        active = [
            self._vectorize_public_active_pokemon(p) for p in pokemon[:controlling]
        ]
        active += [torch.zeros(_ACTIVE_SIZE) for _ in range(controlling - len(active))]
        active = torch.stack(active)
        active = expand_bt(active)

        reserve = [
            self._vectorize_public_reserve_pokemon(p) for p in pokemon[controlling:]
        ]
        num_reserve_padding = 6 - controlling - len(reserve)
        reserve += [torch.zeros(_RESERVE_SIZE) for _ in range(num_reserve_padding)]

        reserve = torch.stack(reserve)
        reserve = expand_bt(reserve)

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
        side_conditions = expand_bt(side_conditions)

        stealthrock = torch.tensor(
            side["sideConditions"].get("stealthrock", [None, 0])[1]
        )
        stealthrock = expand_bt(stealthrock)
        spikes = torch.tensor(side["sideConditions"].get("spikes", [None, 0])[1])
        spikes = expand_bt(spikes)
        toxicspikes = torch.tensor(
            side["sideConditions"].get("toxicspikes", [None, 0])[1]
        )
        toxicspikes = expand_bt(toxicspikes)
        stickyweb = torch.tensor(side["sideConditions"].get("stickyweb", [None, 0])[1])
        stickyweb = expand_bt(stickyweb)

        n = torch.tensor(side["n"])
        n = expand_bt(n)

        total_pokemon = torch.tensor(side["totalPokemon"])
        total_pokemon = expand_bt(total_pokemon)

        faint_counter = torch.tensor(side["faintCounter"])
        faint_counter = expand_bt(faint_counter)

        wisher_slot = (side["wisher"] or {"slot": -1})["slot"]
        wisher_slot = torch.tensor(wisher_slot)
        wisher_slot = expand_bt(wisher_slot)

        return PublicSide(
            n=n,
            total_pokemon=total_pokemon,
            faint_counter=faint_counter,
            active=active,
            reserve=reserve,
            side_conditions=side_conditions,
            stealthrock=stealthrock,
            spikes=spikes,
            toxicspikes=toxicspikes,
            stickyweb=stickyweb,
            wisher=wisher_slot,
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
        return ActivePublicPokemon(
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
        return ReservePublicPokemon(
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
