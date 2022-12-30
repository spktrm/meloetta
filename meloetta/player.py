import re

from typing import Dict, Union, Any, List

from meloetta.client import Client
from meloetta.room import BattleRoom
from meloetta.state import VectorizedState

from bs4 import BeautifulSoup


State = Dict[str, Union[str, Dict[str, Any], List[str]]]


class ChoiceBuilder:
    def __init__(self, room: BattleRoom, state: State):
        self.room = room
        self.state = state
        controls = state["controls"]["controls"]
        self.controls_html = BeautifulSoup(controls, "html.parser")

        self.isMega = any("mega" in choice for choice in state["choice"]["choices"])
        self.isZMove = any("zmove" in choice for choice in state["choice"]["choices"])
        self.isUltraBurst = any(
            "ultra" in choice for choice in state["choice"]["choices"]
        )
        self.isDynamax = any(
            "dynamax" in choice for choice in state["choice"]["choices"]
        )
        self.isTerastal = any(
            "terastal" in choice for choice in state["choice"]["choices"]
        )

    def update(self, room: BattleRoom, state: State):
        self.room = room
        self.state = state

    def get_choices(self):
        teampreview = self.controls_html.find_all(attrs={"name": "chooseTeamPreview"})
        teampreview = [
            (
                choice.attrs.get("data-tooltip"),
                self.room.choose_team_preview,
                [choice.attrs.get("value")],
                {},
            )
            for choice in teampreview
        ]

        reg_moves = self.controls_html.find_all(
            attrs={"name": "chooseMove", "data-tooltip": re.compile(r"^move\|")}
        )
        moves = [
            (
                choice.attrs.get("data-tooltip"),
                self.room.choose_move,
                [choice.attrs.get("value")],
                {
                    "target": choice.attrs.get("data-target"),
                    "isMega": False,
                    "isZMove": False,
                    "isUltraBurst": False,
                    "isDynamax": False,
                    "isTerastal": False,
                },
            )
            for choice in reg_moves
        ]
        if not self.isDynamax:
            max_moves = self.controls_html.find_all(
                attrs={"name": "chooseMove", "data-tooltip": re.compile(r"^maxmove\|")}
            )
            moves += [
                (
                    choice.attrs.get("data-tooltip"),
                    self.room.choose_move,
                    [choice.attrs.get("value")],
                    {
                        "target": choice.attrs.get("data-target"),
                        "isMega": False,
                        "isZMove": False,
                        "isUltraBurst": False,
                        "isDynamax": True,
                        "isTerastal": False,
                    },
                )
                for choice in max_moves
            ]
        z_moves = self.controls_html.find_all(
            attrs={"name": "chooseMove", "data-tooltip": re.compile(r"^zmove\|")}
        )
        moves += [
            (
                choice.attrs.get("data-tooltip"),
                self.room.choose_move,
                [choice.attrs.get("value")],
                {
                    "target": choice.attrs.get("data-target"),
                    "isMega": False,
                    "isZMove": True,
                    "isUltraBurst": False,
                    "isDynamax": False,
                    "isTerastal": False,
                },
            )
            for choice in z_moves
        ]
        t_moves = self.controls_html.find_all(
            attrs={"name": "chooseMove", "data-tooltip": re.compile(r"^terastal\|")}
        )
        moves += [
            (
                choice.attrs.get("data-tooltip"),
                self.room.choose_move,
                [choice.attrs.get("value")],
                {
                    "target": choice.attrs.get("data-target"),
                    "isMega": False,
                    "isZMove": False,
                    "isUltraBurst": False,
                    "isDynamax": False,
                    "isTerastal": True,
                },
            )
            for choice in t_moves
        ]

        move_targets = self.controls_html.find_all(attrs={"name": "chooseMoveTarget"})
        move_targets = [
            (
                choice.attrs.get("data-tooltip"),
                self.room.choose_move_target,
                [choice.attrs.get("value")],
                {},
            )
            for choice in move_targets
        ]

        switches = self.controls_html.find_all(attrs={"name": "chooseSwitch"})
        switches = [
            (
                choice.attrs.get("data-tooltip"),
                self.room.choose_switch,
                [choice.attrs.get("value")],
                {},
            )
            for choice in switches
        ]

        switch_targets = self.controls_html.find_all(
            attrs={"name": "chooseSwitchTarget"}
        )
        switch_targets = [
            (
                choice.attrs.get("data-tooltip"),
                self.room.choose_switch_target,
                [choice.attrs.get("value")],
                {},
            )
            for choice in switch_targets
        ]

        shift = self.controls_html.find_all(attrs={"name": "chooseShift"})
        shift = [
            (
                choice.attrs.get("data-tooltip"),
                self.room.choose_shift,
                [],
                {},
            )
            for choice in shift
        ]

        return teampreview + moves + move_targets + switches + switch_targets + shift


class Player:
    client: Client
    room: BattleRoom
    choice: str
    action_required: bool
    request: Dict[str, Union[str, Dict[str, Any]]]

    @classmethod
    async def create(cls, username, password, address) -> "Player":
        cls = Player()
        cls.client = await Client.create(username, password, address)
        cls.room = BattleRoom()
        cls.action_required = False
        cls.request = None
        cls.started = False
        return cls

    async def recieve(self, data: str):
        if data.startswith(">"):
            try:
                nlIndex = data.index("\n")
            except IndexError:
                nlIndex = -1
            self.room._battle_tag = data[1:nlIndex]
            data = data[nlIndex + 1 :]
        if not data:
            return
        if self.started:
            self.room.recieve(data)
        else:
            if data.startswith("|init|"):
                self.started = True
                self.room.recieve(data)

        state = self.get_state()
        # action_required = state.get("controlsShown", False)

        if "|turn" in data:
            return True

        if "|request" not in data:
            return (state.get("request") or {}).get("forceSwitch")

    def reset(self):
        self.room.reset()
        self.choice = None
        self.request = None

    def get_state(self, raw: bool = False):
        self.state = self.room.get_state(raw=raw)
        return self.state

    def get_vectorized_state(self):
        state = self.room.get_state()
        return VectorizedState.from_battle(self.room, state)

    def get_vectorized_choice(self):
        state = self.room.get_state()
        return VectorizedState.from_battle(self.room, state)

    def get_choices(self):
        return ChoiceBuilder(self.room, self.state).get_choices()

    async def submit_choices(self, msg_list):
        await self.client.send_message(self.room.battle_tag, msg_list)
