from typing import Union, NamedTuple, List, Dict, Any

from meloetta.data import (
    BOOSTS,
    VOLATILES,
    get_status_token,
    get_gender_token,
    get_species_token,
    get_ability_token,
    get_item_token,
    get_move_token,
)


def to_id(string: str):
    return "".join(c for c in string if c.isalnum()).lower()


class Side(NamedTuple):
    n: int
    total_pokemon: int
    faint_counter: int
    side_conditions: Dict[str, Any]
    wisher: Union[Dict[str, Any], None]
    active: List[Dict[str, Any]]
    reserve: List[Dict[str, Any]]


class ReservePokemon(NamedTuple):
    species: str
    slot: int
    hp: int
    fainted: bool
    maxhp: int
    level: int
    gender: int
    moves: List[str]
    ability: str
    base_ability: str
    item: str
    prev_item: str
    terastallized: str
    status: str
    status_stage: int
    last_move: str
    times_attacked: int


class ActivePokemon(NamedTuple):
    species: int
    forme: int
    slot: int
    hp: int
    fainted: bool
    maxhp: int
    level: int
    gender: int
    moves: List[str]
    ability: str
    base_ability: str
    item: str
    prev_item: str
    terastallized: str
    status: str
    status_stage: int
    last_move: str
    times_attacked: int
    boosts: Dict[str, Any]
    volatiles: Dict[str, Any]
    sleep_turns: int
    toxic_turns: int


class Weather(NamedTuple):
    pass


class PseudoWeather(NamedTuple):
    pass


class State(NamedTuple):
    p1: Side
    p2: Side
    weather: str
    pseudo_weather: str
    turn: int
    last_move: str
    log: List[str]


class VectorizedState:
    def __init__(self, battle, state):
        self.battle = battle
        self.state = state
        self.gen = self.state["dex"]["gen"]

    @classmethod
    def from_battle(self, battle, state):
        vstate = VectorizedState(battle, state)
        return vstate.vectorize()

    def vectorize(self):
        p1, p2 = self._vectorize_sides()
        return State(
            p1=p1,
            p2=p2,
            last_move=self.state["lastMove"],
            weather=self.state["weather"],
            pseudo_weather=self.state["pseudoWeather"],
            turn=self.state["turn"],
            log=self.state["stepQueue"],
        )

    def _vectorize_sides(self):
        p1 = self._vectorize_side("mySide")
        p2 = self._vectorize_side("farSide")
        return p1, p2

    def _vectorize_side(self, side_id: str):
        side = self.state[side_id]
        return Side(
            n=side["n"],
            total_pokemon=side["totalPokemon"],
            faint_counter=side["faintCounter"],
            active=[self._vectorize_public_active_pokemon(p) for p in side["active"]],
            reserve=[
                self._vectorize_public_reserve_pokemon(p) for p in side["pokemon"]
            ],
            side_conditions=side["sideConditions"],
            wisher=side["wisher"],
        )

    def _vectorize_public_active_pokemon(self, pokemon):
        if pokemon is None:
            return None

        moves = [
            (get_move_token(self.gen, "name", move), pp)
            for move, pp in pokemon["moveTrack"]
        ]
        boosts = [pokemon["boosts"].get(boost, 0) for boost in BOOSTS]
        volatiles = [pokemon["volatiles"].get(volatile, 0) for volatile in VOLATILES]
        forme = pokemon["speciesForme"].replace(pokemon["name"] + "-", "")
        return ActivePokemon(
            species=get_species_token(self.gen, "name", pokemon["name"]),
            forme=get_species_token(self.gen, "forme", forme),
            slot=pokemon["slot"],
            hp=pokemon["hp"],
            maxhp=pokemon["maxhp"],
            fainted=pokemon["fainted"],
            level=pokemon["level"],
            gender=get_gender_token(pokemon["gender"]),
            moves=moves,
            ability=get_ability_token(self.gen, "name", pokemon["ability"]),
            base_ability=get_ability_token(self.gen, "name", pokemon["baseAbility"]),
            item=get_item_token(self.gen, "name", pokemon["item"]),
            prev_item=get_item_token(self.gen, "name", pokemon["prevItem"]),
            terastallized=pokemon["terastallized"],
            status=get_status_token(pokemon["status"]),
            status_stage=pokemon["statusStage"],
            last_move=get_move_token(self.gen, "name", pokemon["lastMove"]),
            times_attacked=pokemon["timesAttacked"],
            boosts=boosts,
            volatiles=volatiles,
            sleep_turns=pokemon["statusData"]["sleepTurns"],
            toxic_turns=pokemon["statusData"]["toxicTurns"],
        )

    def _vectorize_public_reserve_pokemon(self, pokemon):
        return ReservePokemon(
            species=pokemon["speciesForme"],
            slot=pokemon["slot"],
            hp=pokemon["hp"],
            maxhp=pokemon["maxhp"],
            fainted=pokemon["fainted"],
            level=pokemon["level"],
            gender=pokemon["gender"],
            moves=pokemon["moveTrack"],
            ability=pokemon["ability"],
            base_ability=pokemon["baseAbility"],
            item=pokemon["item"],
            prev_item=pokemon["prevItem"],
            terastallized=pokemon["terastallized"],
            status=pokemon["status"],
            status_stage=pokemon["statusStage"],
            last_move=pokemon["lastMove"],
            times_attacked=pokemon["timesAttacked"],
        )

    def _vectorize_weather(self):
        return

    def _vectorize_pseudo_weather(self):
        return self.state["pseudoWeather"]
