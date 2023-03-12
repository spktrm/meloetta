import torch
import numpy as np

from typing import Tuple, Dict, Any

from meloetta.room import BattleRoom

from meloetta.frameworks.porygon import ReplayBuffer
from meloetta.frameworks.porygon.model import (
    PorygonModel,
    ModelOutput,
    PostProcess,
)

from meloetta.actors.base import Actor
from meloetta.actors.types import State, Choices, Battle


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


class PorygonActor(Actor):
    def __init__(
        self,
        model: PorygonModel,
        replay_buffer: ReplayBuffer = None,
    ):
        self.model = model
        self.gen = model.gen

        self.replay_buffer = replay_buffer
        self.hidden_state = model.core.initial_state(1)

        if replay_buffer is not None:
            self.buffer_index = self.replay_buffer._get_index()
        self.step_index = 0

        self.env_outputs = []
        self.model_outputs = []

    @property
    def storing_transition(self):
        return self.replay_buffer is not None

    def choose_action(
        self,
        env_output: State,
        room: BattleRoom,
        choices: Choices,
    ):
        output: Tuple[ModelOutput, PostProcess]
        with torch.no_grad():
            output = self.model(env_output, self.hidden_state, choices)
        model_output, postprocess, self.hidden_state = output

        if self.storing_transition:
            self._store_transition(env_output, model_output, room)

        data = postprocess.data
        index = postprocess.index
        func, args, kwargs = data[index.item()]
        return func, args, kwargs

    def _clean_env_output(self, env_output: State):
        return {
            k: env_output[k]
            for k in self.model.state_fields
            if isinstance(env_output[k], torch.Tensor)
        }

    def _clean_model_output(self, model_output: ModelOutput):
        to_store = {}
        to_store.update(
            {
                k: v.squeeze()
                for k, v in model_output.indices._asdict().items()
                if isinstance(v, torch.Tensor)
            }
        )
        to_store.update(
            {
                k: v.squeeze()
                for k, v in model_output._asdict().items()
                if isinstance(v, torch.Tensor)
            }
        )
        return to_store

    def _store_transition(
        self, env_output: State, model_output: ModelOutput, room: BattleRoom
    ):
        env_output = self._clean_env_output(env_output)
        model_output = self._clean_model_output(model_output)

        self.env_outputs.append(env_output)
        self.model_outputs.append(model_output)

    def _populate_buffer(self) -> int:
        for step_index, (env_output, model_output) in enumerate(
            zip(self.env_outputs[1:], self.model_outputs)
        ):
            to_store = {**env_output, **model_output}

            self.replay_buffer.store_sample(self.buffer_index, step_index, to_store)

        return step_index

    def post_match(self, room: BattleRoom):
        if self.storing_transition:
            datum = room.get_reward()
            reward = datum["reward"]

            final_turn = self._populate_buffer()

            self.replay_buffer.append_reward(self.buffer_index, final_turn, reward)

    def _vectorize_public_pokemon(
        self, pokemon: Dict[str, Any], active: bool
    ) -> torch.Tensor:
        return pokemon

    def _vectorize_private_pokemon(self, pokemon: Dict[str, Any]) -> torch.Tensor:
        moves = pokemon["moves"]
        return torch.tensor(
            [
                get_species_token(self.gen, "name", pokemon["name"]),
                get_species_token(self.gen, "forme", pokemon["speciesForme"]),
                pokemon["slot"],
                pokemon["hp"],
                pokemon["maxhp"],
                pokemon["hp"] / pokemon["maxhp"],
                1 if pokemon["fainted"] else 0,
                pokemon["level"],
                get_gender_token(pokemon["gender"]),
                get_ability_token(self.gen, "name", pokemon["ability"]),
                get_ability_token(self.gen, "name", pokemon["baseAbility"]),
                get_item_token(self.gen, "name", pokemon["item"]),
                get_item_token(self.gen, "name", pokemon["prevItem"]),
                get_item_effect_token(pokemon["itemEffect"]),
                get_item_effect_token(pokemon["prevItemEffect"]),
                get_status_token(pokemon["status"]),
                pokemon["statusStage"],
                min(pokemon["statusData"]["sleepTurns"], 3),
                min(pokemon["statusData"]["toxicTurns"], 15),
                get_move_token(self.gen, "id", pokemon["lastMove"]),
                get_move_token(self.gen, "id", moves[min(0, len(moves) - 1)]),
                get_move_token(self.gen, "id", moves[min(1, len(moves) - 1)]),
                get_move_token(self.gen, "id", moves[min(2, len(moves) - 1)]),
                get_move_token(self.gen, "id", moves[min(3, len(moves) - 1)]),
                get_type_token(self.gen, pokemon["terastallized"]),
                min(pokemon["timesAttacked"], 6),
            ]
        )

    def get_vectorized_state(
        self, room: BattleRoom, battle: Battle
    ) -> Dict[str, torch.Tensor]:
        my_private_side = {p["searchid"]: p for p in battle["myPokemon"]}

        my_public_side = {p["searchid"]: p for p in battle["mySide"]["pokemon"]}
        my_active = {p["searchid"] for p in battle["mySide"]["active"]}

        opp_public_side = {p["searchid"]: p for p in battle["farSide"]["pokemon"]}
        opp_active = {p["searchid"] for p in battle["farSide"]["active"]}

        boosts = []
        volatiles = []

        for public_side, active_list in zip(
            [my_public_side, opp_public_side], [my_active, opp_active]
        ):
            for pokemon in public_side.values():
                if pokemon["searchid"] in active_list:
                    boosts_vector = torch.tensor(
                        [pokemon["boosts"].get(boost, 0) for boost in BOOSTS]
                    )
                    boosts.append(boosts_vector)
                    volatile_vector = torch.tensor(
                        [
                            1 if volatile in pokemon["volatiles"] else 0
                            for volatile in VOLATILES
                        ]
                    )
                    volatiles.append(volatile_vector)

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
                sc_datum.get(side_condition, [None, 0, -1, -1])[1:]
                for side_condition in SIDE_CONDITIONS
                if side_condition
                not in {"stealthrock", "spikes", "toxicspikes", "stickyweb"}
            ]
            side_conditions_vector = [i for o in side_conditions_vector for i in o]
            side_conditions_vector += [sc_datum.get("stealthrock", [None, 0])[1]]
            side_conditions_vector += [sc_datum.get("toxicspikes", [None, 0])[1]]
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
        for sid, p in my_private_side.items():
            public_data = my_public_side.get(sid, {})
            for key, value in p:
                if key 
                    public_data[key] = value
            private_vector = self._vectorize_private_pokemon(private_data)

        my_public_side_vectors = torch.stack(
            [
                self._vectorize_public_pokemon(
                    {
                        **p,
                        **my_private_side.get(sid, {}),
                    }
                )
                for sid, p in my_public_side.items()
            ]
        )
        opp_public_side_vectors = torch.stack(
            [self._vectorize_public_pokemon(p) for sid, p in opp_public_side.items()]
        )

        return VectorizedState.from_battle(room, battle).to_dict()
