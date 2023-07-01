import time
import torch
import torch.nn.functional as F

from typing import Dict, Any, List

from torch.nn.utils.rnn import pad_sequence

from meloetta.room import BattleRoom

from meloetta.frameworks.nash_ketchum.buffer import ReplayBuffer
from meloetta.frameworks.nash_ketchum.modelv2 import NAshKetchumModel

from meloetta.data import to_id
from meloetta.actors.base import Actor
from meloetta.actors.types import State, Choices, Battle, TensorDict
from meloetta.utils import expand_bt
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


SIDES = ["mySide", "farSide"]
KEYS = {
    "terastallized",
    "prevItemEffect",
    # "reviving",
    # "boosts",
    # "slot",
    "speciesForme",
    # "ident",
    "lastMove",
    # "movestatuses",
    # "hpcolor",
    # "details",
    "baseAbility",
    "name",
    "fainted",
    # "shiny",
    "stats",
    "timesAttacked",
    "hp",
    # "side",
    "statusData",
    # "volatiles",
    # "searchid",
    # "pokeball",
    "active",
    "gender",
    "item",
    # "condition",
    "moveTrack",
    # "sprite",
    "moves",
    "maxhp",
    "teraType",
    # "canGmax",
    # "commanding",
    "status",
    # "turnstatuses",
    "itemEffect",
    "ability",
    "level",
    "prevItem",
    # "statusStage",
}

POKEMON_VECTOR_SIZE = 38


class NAshKetchumActor(Actor):
    def __init__(
        self,
        model: NAshKetchumModel = None,
        replay_buffer: ReplayBuffer = None,
        pid: str = None,
    ):
        self.model = model
        self.gen = model.gen

        self.replay_buffer = replay_buffer

        self.step_index = 0
        self.index = None

        self.turn = 0
        self.pid = pid
        self.turns = []
        self.trajectory = []

    @property
    def storing_transition(self):
        return self.replay_buffer is not None

    @torch.no_grad()
    def choose_action(self, state: State, room: BattleRoom, choices: Choices):
        model_output = self.model.forward(
            state, compute_log_policy=False, compute_value=False
        )
        postprocess = self.model.postprocess(
            state=state,
            model_output=model_output,
            choices=choices,
        )

        data = postprocess.data
        index = postprocess.index
        func, args, kwargs = data[index]

        if self.storing_transition:
            action_type = model_output["action_type_index"].item()
            if action_type == 0:
                policies_to_store = []
                for policy_select, policy_mask in (
                    (0, state["action_type_mask"]),
                    (1, state["flag_mask"]),
                    (2, state["move_mask"]),
                ):
                    if (policy_mask.sum() > 1).item():
                        policies_to_store.append(policy_select)
                for i, policy_select in enumerate(policies_to_store):
                    model_output["policy_select"] = torch.tensor(
                        policy_select, dtype=torch.long
                    )
                    model_output["utc"] = torch.tensor(
                        self.turn + i / 100 + self.pid / 2, dtype=torch.float64
                    )
                    self.store_transition(state, model_output, room)
            elif action_type == 1:
                policies_to_store = []
                for policy_select, policy_mask in (
                    (0, state["action_type_mask"]),
                    (3, state["switch_mask"]),
                ):
                    if (policy_mask.sum() > 1).item():
                        policies_to_store.append(policy_select)
                for i, policy_select in enumerate([0, 3]):
                    model_output["policy_select"] = torch.tensor(
                        policy_select, dtype=torch.long
                    )
                    model_output["utc"] = torch.tensor(
                        self.turn + i / 100 + self.pid / 2, dtype=torch.float64
                    )
                    self.store_transition(state, model_output, room)
            self.turn += 1

        return func, args, kwargs

    def get_index(self, battle_tag: str):
        if self.index is None:
            self.index = self.replay_buffer._get_index(battle_tag, self.pid)
        return self.index

    def store_transition(
        self, state: State, model_output: TensorDict, room: BattleRoom
    ):
        to_store = self.model.clean(state)
        to_store = {**to_store, **model_output}
        to_store = {
            k: v
            for k, v in to_store.items()
            if k in self.replay_buffer.buffers.keys() and v is not None
        }
        self.trajectory.append(to_store)

    def post_match(self, room: BattleRoom):
        if self.storing_transition:

            def _prepare_trajectory(trajectory: List[TensorDict]):
                return {
                    key: torch.stack([step[key] for step in trajectory]).squeeze()
                    for key in trajectory[0].keys()
                }

            trajectory = _prepare_trajectory(self.trajectory)
            self.replay_buffer.store_trajectory(
                self.get_index(room.battle_tag), self.pid, trajectory
            )

            datum = room.get_reward()

            battle_tag = room.battle_tag
            index = self.get_index(battle_tag)

            self.replay_buffer.append_reward(index, -1, self.pid, datum["reward"])
            self.replay_buffer.register_done(battle_tag, self.pid)

    def _vectorize_pokemon(
        self, pokemon: Dict[str, Any], sideid: int, public: bool = False
    ) -> torch.Tensor:
        if public:
            moves = {}
        else:
            moves = {
                get_move_token(self.gen, "id", move): -1 for move in pokemon["moves"]
            }

        for move, pp in pokemon.get("moveTrack", []):
            token = get_move_token(self.gen, "id", move)
            moves[token] = pp

        move_keys = list(moves.keys())[:4]
        move_values = list(moves.values())[:4]

        move_keys += [-1 for _ in range(4 - len(move_keys))]
        move_values += [-1 for _ in range(4 - len(move_values))]

        hp = pokemon.get("hp", 0) * (not public)
        maxhp = pokemon.get("maxhp", 0) * (not public)
        hp_ratio = pokemon.get("hp", 0) / max(pokemon.get("maxhp", 0), 1)
        fainted = hp_ratio == 0

        last_move_token = get_move_token(self.gen, "id", pokemon.get("lastMove"))

        stats = pokemon.get("stats", {})
        status_data = pokemon.get("statusData", {})

        data = [
            get_species_token(self.gen, "name", pokemon["name"]),
            get_species_token(self.gen, "forme", pokemon["speciesForme"]),
            pokemon.get("slot", 0),
            hp,
            maxhp,
            hp_ratio,
            stats.get("atk", -1),
            stats.get("def", -1),
            stats.get("spa", -1),
            stats.get("spd", -1),
            stats.get("spe", -1),
            fainted,
            pokemon.get("active", 0),
            pokemon.get("level", 1),
            get_gender_token(pokemon.get("gender", "")),
            get_ability_token(self.gen, "name", pokemon["ability"]),
            get_ability_token(self.gen, "name", pokemon["baseAbility"]),
            get_item_token(self.gen, "name", pokemon.get("item", "")),
            get_item_token(self.gen, "name", pokemon.get("prevItem", "")),
            get_item_effect_token(pokemon.get("itemEffect", "")),
            get_item_effect_token(pokemon.get("prevItemEffect", "")),
            get_status_token(pokemon.get("status", "")),
            min(status_data.get("sleepTurns", 0), 3),
            min(status_data.get("toxicTurns", 0), 15),
            last_move_token,
            moves.get(last_move_token, 0),
            *move_keys,
            *move_values,
            get_type_token(self.gen, pokemon.get("terastallized")),
            get_type_token(self.gen, pokemon.get("teraType")),
            min(pokemon.get("timesAttacked", 0), 6),
            sideid,
        ]
        return torch.tensor(data)

    def _get_turns(
        self,
        my_private_side: Dict[str, Dict[str, Any]],
        opp_public_side: Dict[str, Dict[str, Any]],
        turns: List[List[str]],
        my_sideid: str,
    ) -> torch.Tensor:
        def _cleanse_ident(string: str):
            for targ in ["p1", "p2"]:
                string = string.replace(f"{targ}a", targ)
            return string

        def _find_index(lst: List[str], query: str) -> int:
            for idx, string in enumerate(lst):
                if query in string:
                    return idx

        my_private_side_keys, my_private_side_values = list(
            zip(*my_private_side.items())
        )
        opp_public_side_keys, opp_public_side_values = list(
            zip(*opp_public_side.items())
        )

        action_history = []

        for turn in turns:
            turn_vectors = []

            for line in reversed(turn):
                _, action_type, *args = line.split("|")

                if action_type == "move":
                    action_token = 0
                    ident = _cleanse_ident(args[0])
                    move_to_id = to_id(args[1])

                    if f"{my_sideid}a" in args[0]:
                        switch_index = _find_index(my_private_side_keys, ident)
                        try:
                            move_index = my_private_side_values[switch_index][
                                "moves"
                            ].index(move_to_id)
                        except:
                            move_index = -1
                        player_id = 0

                    else:
                        switch_index = _find_index(opp_public_side_keys, ident)
                        move_lst = [
                            to_id(move)
                            for move, _ in opp_public_side_values[switch_index][
                                "moveTrack"
                            ]
                        ]
                        try:
                            move_index = move_lst.index(move_to_id)
                        except:
                            move_index = -1
                        player_id = 1

                elif action_type == "switch":
                    action_token = 1
                    move_index = -1
                    ident = _cleanse_ident(args[0])

                    if f"{my_sideid}a" in args[0]:
                        switch_index = _find_index(my_private_side_keys, ident)
                        player_id = 0

                    else:
                        switch_index = _find_index(opp_public_side_keys, ident)
                        player_id = 1

                else:
                    continue

                turn_vector = torch.tensor(
                    [player_id, action_token, move_index, switch_index],
                    dtype=torch.long,
                )
                turn_vectors.append(turn_vector)

            if turn_vectors:
                turn_vectors = torch.stack(turn_vectors[:4])
                turn_vectors = F.pad(
                    turn_vectors, (0, 0, 0, 4 - turn_vectors.shape[0]), value=-1
                )
            else:
                turn_vectors = -torch.ones(4, 4)
            action_history.append(turn_vectors)

        action_history = torch.stack(action_history)
        action_history = F.pad(
            action_history, (0, 0, 0, 0, 0, 10 - action_history.shape[0]), value=-1
        )

        return action_history

    def get_vectorized_state(
        self, room: BattleRoom, battle: Battle
    ) -> Dict[str, torch.Tensor]:
        my_private_side = {p["searchid"]: p for p in battle.get("myPokemon") or []}

        my_public_side = {
            p["searchid"]: p for p in battle["mySide"]["pokemon"] if p is not None
        }
        my_active = {p["searchid"] for p in battle["mySide"]["active"] if p is not None}

        opp_public_side = {
            p["searchid"]: p for p in battle["farSide"]["pokemon"] if p is not None
        }
        opp_active = {
            p["searchid"] for p in battle["farSide"]["active"] if p is not None
        }

        my_public_side_lst = [
            (k, v) if k in my_public_side else ("", None)
            for k, v in my_private_side.items()
        ]

        # self.turns += room.get_turns(1)
        # turn_vectors = self._get_turns(
        #     my_private_side,
        #     opp_public_side,
        #     self.turns[-10:],
        #     battle["mySide"]["sideid"],
        # )

        boosts = [None, None]
        volatiles = [None, None]

        for sid, (public_side, active_list) in enumerate(
            zip([my_public_side, opp_public_side], [my_active, opp_active])
        ):
            for pokemon in public_side.values():
                if pokemon["searchid"] in active_list:
                    boosts_vector = torch.tensor(
                        [pokemon["boosts"].get(boost, 0) for boost in BOOSTS]
                    )
                    boosts[sid] = boosts_vector

                    volatile_vector = torch.tensor(
                        [
                            1 if volatile in pokemon["volatiles"] else 0
                            for volatile in VOLATILES
                        ]
                    )
                    volatiles[sid] = volatile_vector

        for sid in range(len(boosts)):
            if boosts[sid] is None:
                boosts[sid] = torch.zeros(len(BOOSTS), dtype=torch.long)

        for sid in range(len(volatiles)):
            if volatiles[sid] is None:
                volatiles[sid] = torch.zeros(len(VOLATILES), dtype=torch.long)

        boosts = torch.stack(boosts)
        volatiles = torch.stack(volatiles)

        side_conditions = []
        n = []
        total_pokemon = []
        faint_counter = []
        wisher_slot = []

        for side in SIDES:
            side_datum = battle[side]
            sc_datum = battle[side]["sideConditions"]

            side_conditions_vector = [
                sc_datum.get(side_condition, [None, 0, 0, 0])[1:]
                for side_condition in SIDE_CONDITIONS
                if side_condition
                not in {"stealthrock", "spikes", "toxicspikes", "stickyweb"}
            ]
            side_conditions_vector = [i for o in side_conditions_vector for i in o]
            side_conditions_vector += [sc_datum.get("stealthrock", [None, 0])[1]]
            side_conditions_vector += [sc_datum.get("toxicspikes", [None, 0])[1]]
            side_conditions_vector += [sc_datum.get("spikes", [None, 0])[1]]
            side_conditions_vector += [sc_datum.get("stickyweb", [None, 0])[1]]
            side_conditions += [torch.tensor(side_conditions_vector)]

            n += [torch.tensor(side_datum["n"])]
            total_pokemon += [torch.tensor(side_datum["totalPokemon"])]
            faint_counter += [torch.tensor(min(6, side_datum["faintCounter"]))]
            wisher_slot += [
                torch.tensor((side_datum["wisher"] or {"slot": -1})["slot"])
            ]

        side_conditions = torch.stack(side_conditions)
        n = torch.stack(n)
        total_pokemon = torch.stack(total_pokemon)
        faint_counter = torch.stack(faint_counter)
        wisher_slot = torch.stack(wisher_slot)

        my_private_side_vectors = []
        for sid, private_data in my_private_side.items():
            datum = {}

            public_data = my_public_side.get(sid, {})

            datum["active"] = sid in my_active
            for key in KEYS:
                value = private_data.get(key) or public_data.get(key)
                if value is not None:
                    datum[key] = value

            private_vector = self._vectorize_pokemon(datum, sideid=0)
            my_private_side_vectors.append(private_vector)

        my_public_side_vectors = []
        for sid, public_data in my_public_side_lst:
            if not sid or public_data is None:
                public_vector = -torch.ones(POKEMON_VECTOR_SIZE)

            else:
                datum = {}
                datum["active"] = sid in my_active
                for key in KEYS:
                    if key in public_data:
                        value = public_data.get(key)
                        if value is not None:
                            datum[key] = value

                public_vector = self._vectorize_pokemon(datum, sideid=0, public=True)
            my_public_side_vectors.append(public_vector)

        opp_public_side_vectors = []
        for sid, public_data in opp_public_side.items():
            datum = {}
            datum["active"] = sid in opp_active
            for key in KEYS:
                if key in public_data:
                    value = public_data.get(key)
                    if value is not None:
                        datum[key] = value

            public_vector = self._vectorize_pokemon(datum, sideid=1, public=True)
            opp_public_side_vectors.append(public_vector)

        if my_private_side_vectors:
            my_private_side = torch.stack(my_private_side_vectors)

            padding_length = 12 - my_private_side.shape[0]
            my_private_side_padding = -torch.ones(POKEMON_VECTOR_SIZE).expand(
                padding_length, -1
            )
            my_private_side_padding[-padding_length:] *= 2
            my_private_side_vectors = torch.cat(
                (my_private_side, my_private_side_padding)
            )
        else:
            my_private_side_vectors = -torch.ones(12, POKEMON_VECTOR_SIZE)
            my_private_side_vectors = my_private_side_vectors

        if my_public_side_vectors:
            my_public_side_vectors = torch.stack(my_public_side_vectors)
        else:
            my_public_side_vectors = -torch.ones(12, POKEMON_VECTOR_SIZE)

        opp_public_side_vectors = torch.stack(opp_public_side_vectors)

        sides = pad_sequence(
            [
                my_private_side_vectors,
                my_public_side_vectors,
                F.pad(
                    opp_public_side_vectors,
                    (0, 0, 0, 6 - opp_public_side_vectors.shape[0]),
                    value=-1,
                ),
            ],
            batch_first=True,
            padding_value=-2,
        )

        pseudoweathers = {
            (pseudoweather[0]): pseudoweather[1:]
            for pseudoweather in battle["pseudoWeather"]
        }
        pseudoweathers = [
            torch.tensor(pseudoweathers.get(pseudoweather, [-1, -1]))
            for pseudoweather in PSEUDOWEATHERS
        ]
        pseudoweathers = torch.stack(pseudoweathers)

        weather = torch.tensor(
            [
                get_weather_token(battle["weather"]),
                battle["weatherTimeLeft"],
                battle["weatherMinTimeLeft"],
            ]
        )

        turn = torch.tensor([battle["turn"]])
        scalars = torch.cat((turn, n, total_pokemon, faint_counter), dim=-1).long()

        turn_vectors = -torch.ones(10, 4, 4)
        state = {
            "sides": sides,
            "boosts": boosts,
            "volatiles": volatiles,
            "weather": weather,
            "side_conditions": side_conditions,
            "pseudoweathers": pseudoweathers,
            "wisher": wisher_slot,
            "scalars": scalars,
            "hist": turn_vectors,
        }
        state = {key: expand_bt(value) for key, value in state.items()}
        return state
