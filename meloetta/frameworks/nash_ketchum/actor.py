import json
import time
import torch
import traceback

from typing import Dict, Any

from torch.nn.utils.rnn import pad_sequence

from meloetta.room import BattleRoom

from meloetta.frameworks.nash_ketchum import ReplayBuffer
from meloetta.frameworks.nash_ketchum.model import (
    NAshKetchumModel,
    ModelOutput,
    PostProcess,
)


from meloetta.actors.base import Actor
from meloetta.actors.types import State, Choices, Battle
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


class NAshKetchumActor(Actor):
    def __init__(
        self,
        model: NAshKetchumModel,
        replay_buffer: ReplayBuffer = None,
        pid: str = None,
    ):
        self.model = model
        self.gen = model.gen

        self.replay_buffer = replay_buffer

        self.step_index = 0
        self.index = None

        self.env_outputs = []
        self.model_outputs = []

        self.pid = pid

    @property
    def storing_transition(self):
        return self.replay_buffer is not None

    @torch.no_grad()
    def choose_action(
        self,
        state: State,
        room: BattleRoom,
        choices: Choices,
    ):
        model_output: ModelOutput
        postprocess: PostProcess

        try:
            model_output = self.model.acting_forward(state)
            postprocess = self.model.postprocess(
                state=state,
                model_output=model_output,
                choices=choices,
            )

            data = postprocess.data
            index = postprocess.index
            func, args, kwargs = data[index]
        except:
            trace = traceback.format_exc()
            print(trace)

            battle_tag = room.battle_tag

            json_state = room.get_state()
            side = json_state["side"]

            datum = {
                "json_state": json_state,
                "choices": {
                    k: {sk: sv[1:] for sk, sv in v.items()} for k, v in choices.items()
                },
                "indices": model_output.indices.to_json(),
                "policy": model_output.pi.to_json(),
                "logits": model_output.logit.to_json(),
                "traceback": trace.split("\n"),
                "state_dict": self.model.state_dict(),
                "state": state,
            }
            torch.save(datum, f"errors/{battle_tag}-{side}-model.pt")

            return self.choose_action(state, room, choices)
        else:
            if self.storing_transition:
                self.store_transition(state, model_output, room)

        return func, args, kwargs

    def get_index(self, battle_tag: str):
        if self.index is None:
            self.index = self.replay_buffer._get_index(battle_tag)
        return self.index

    def store_transition(
        self, state: State, model_output: ModelOutput, room: BattleRoom
    ):
        state = self.model.clean(state)
        to_store = model_output.to_store(state)
        index = self.get_index(room.battle_tag)
        self.replay_buffer.store_sample(index, to_store, self.pid)

    def post_match(self, room: BattleRoom):
        if self.storing_transition:
            datum = room.get_reward()
            index = self.get_index(room.battle_tag)

            self.replay_buffer.append_reward(index, self.pid, datum["reward"])
            self.replay_buffer.register_done(room.battle_tag)

    def _vectorize_pokemon(
        self, pokemon: Dict[str, Any], public: bool = False
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

        data = [
            get_species_token(self.gen, "name", pokemon["name"]),
            get_species_token(self.gen, "forme", pokemon["speciesForme"]),
            pokemon.get("slot", 0),
            hp,
            maxhp,
            hp_ratio,
            pokemon.get("stats", {}).get("atk", -1),
            pokemon.get("stats", {}).get("def", -1),
            pokemon.get("stats", {}).get("spa", -1),
            pokemon.get("stats", {}).get("spd", -1),
            pokemon.get("stats", {}).get("spe", -1),
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
            min(pokemon.get("statusData", {}).get("sleepTurns", 0), 3),
            min(pokemon.get("statusData", {}).get("toxicTurns", 0), 15),
            last_move_token,
            moves.get(last_move_token, 0),
            *move_keys,
            *move_values,
            get_type_token(self.gen, pokemon.get("terastallized")),
            get_type_token(self.gen, pokemon.get("teraType")),
            min(pokemon.get("timesAttacked", 0), 6),
        ]
        return torch.tensor(data)

    def get_vectorized_state(
        self, room: BattleRoom, battle: Battle
    ) -> Dict[str, torch.Tensor]:
        my_private_side = {p["searchid"]: p for p in battle["myPokemon"]}

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
        for i, (sid, private_data) in enumerate(my_private_side.items()):
            datum = {}

            public_data = my_public_side.get(sid, {})
            if i == 0:
                keys = list(public_data) + list(private_data)

            datum["active"] = sid in my_active
            for key in keys:
                value = private_data.get(key) or public_data.get(key)
                if value is not None:
                    datum[key] = value

            private_vector = self._vectorize_pokemon(datum)
            my_private_side_vectors.append(private_vector)

        my_public_side_vectors = []
        for i, (sid, public_data) in enumerate(my_public_side.items()):
            datum = {}
            datum["active"] = sid in my_active
            for key in keys:
                if key in public_data:
                    value = public_data.get(key)
                    if value is not None:
                        datum[key] = value

            public_vector = self._vectorize_pokemon(datum, public=True)
            my_public_side_vectors.append(public_vector)

        opp_public_side_vectors = []
        for i, (sid, public_data) in enumerate(opp_public_side.items()):
            datum = {}
            datum["active"] = sid in opp_active
            for key in keys:
                if key in public_data:
                    value = public_data.get(key)
                    if value is not None:
                        datum[key] = value

            public_vector = self._vectorize_pokemon(datum, public=True)
            opp_public_side_vectors.append(public_vector)

        my_private_side = torch.stack(my_private_side_vectors)
        my_private_side = torch.cat(
            (my_private_side, -torch.ones_like(my_private_side))
        )

        my_public_side = torch.stack(my_public_side_vectors)
        opp_public_side = torch.stack(opp_public_side_vectors)

        sides = pad_sequence(
            [my_private_side, my_public_side, opp_public_side],
            batch_first=True,
            padding_value=-1,
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

        state = {
            "sides": sides,
            "boosts": boosts,
            "volatiles": volatiles,
            "weather": weather,
            "side_conditions": side_conditions,
            "pseudoweathers": pseudoweathers,
            "wisher": wisher_slot,
            "scalars": scalars,
        }
        state = {key: expand_bt(value) for key, value in state.items()}
        return state
