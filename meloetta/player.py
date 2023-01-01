import re

from typing import Dict, Union, Any, List

from meloetta.client import Client
from meloetta.room import BattleRoom

from bs4 import BeautifulSoup


State = Dict[str, Union[str, Dict[str, Any], List[str]]]


class ChoiceBuilder:
    def __init__(self, room: BattleRoom):
        self.room = room
        tier = room.get_js_attr("battle.tier")
        self.gen = int(re.search(r"([0-9])", tier).group())
        self.gametype = room.get_js_attr("battle.gameType")

        controls = room.get_js_attr("controls.controls")
        choices = room.get_js_attr("choice.choices")
        choices = [c for c in choices if c is not None]
        self.pos = len(choices)

        self.html = controls
        self.soup = BeautifulSoup(controls, "html.parser")

        self.isMega = any("mega" in choice for choice in choices)
        self.isZMove = any("zmove" in choice for choice in choices)
        self.isUltraBurst = any("ultra" in choice for choice in choices)
        self.isDynamax = any("dynamax" in choice for choice in choices)
        self.isTerastal = any("terastal" in choice for choice in choices)

        choices = []
        self.checkboxes = {
            checkbox.find("input").attrs["name"]
            for checkbox in self.soup.find_all("label")
        }

    def get_choices(self):
        choices = []
        choices += self.get_teampreview()
        choices += self.get_moves()
        choices += self.get_move_targets()
        choices += self.get_switches()
        choices += self.get_switch_targets()
        if self.gen == 8:
            choices += self.get_max_moves()
        if self.gen == 9:
            choices += self.get_tera_moves()
        if self.gen == 7:
            choices += self.get_zmoves()
        if self.gen == 7 or self.gen == 6:
            choices += self.get_mega()
        if self.gametype == "triples" and self.pos != 1:
            choices += self.get_shifts()
        return choices

    def get_teampreview(self):
        teampreview = self.soup.find_all(attrs={"name": "chooseTeamPreview"})
        return [
            (
                choice.attrs.get("data-tooltip"),
                self.room.choose_team_preview,
                [choice.attrs.get("value")],
                {},
            )
            for choice in teampreview
        ]

    def get_moves(self):
        self.reg_moves = self.soup.find_all(
            attrs={"name": "chooseMove", "data-tooltip": re.compile(r"^move\|")}
        )
        return [
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
            for choice in self.reg_moves
        ]

    def get_max_moves(self):
        max_moves = []
        if not self.isDynamax:
            max_moves = self.soup.find_all(
                attrs={"name": "chooseMove", "data-tooltip": re.compile(r"^maxmove\|")}
            )
            max_moves = [
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
        return max_moves

    def get_mega(self):
        if not self.isMega and "megaevo" in self.checkboxes:
            mega_moves = self.soup.find_all(attrs={"name": "chooseMove"})
            return [
                (
                    choice.attrs.get("data-tooltip"),
                    self.room.choose_move,
                    [choice.attrs.get("value")],
                    {
                        "target": choice.attrs.get("data-target"),
                        "isMega": True,
                        "isZMove": False,
                        "isUltraBurst": False,
                        "isDynamax": False,
                        "isTerastal": False,
                    },
                )
                for choice in mega_moves
            ]

    def get_zmoves(self):
        z_moves = []
        if not self.isZMove and "zmove" in self.checkboxes:
            z_moves = self.soup.find_all(
                attrs={"name": "chooseMove", "data-tooltip": re.compile(r"^zmove\|")}
            )
            z_moves = [
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
            return z_moves

    def get_tera_moves(self):
        tera_moves = []
        if not self.isTerastal and "terastallize" in self.checkboxes:
            tera_moves = [
                (
                    choice.attrs.get("data-tooltip") + " tera",
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
                for choice in self.reg_moves
            ]
        return tera_moves

    def get_move_targets(self):
        move_targets = self.soup.find_all(attrs={"name": "chooseMoveTarget"})
        return [
            (
                choice.attrs.get("data-tooltip"),
                self.room.choose_move_target,
                [choice.attrs.get("value")],
                {},
            )
            for choice in move_targets
        ]

    def get_switches(self):
        switches = self.soup.find_all(attrs={"name": "chooseSwitch"})
        return [
            (
                choice.attrs.get("data-tooltip"),
                self.room.choose_switch,
                [choice.attrs.get("value")],
                {},
            )
            for choice in switches
        ]

    def get_switch_targets(self):
        switch_targets = self.soup.find_all(attrs={"name": "chooseSwitchTarget"})
        return [
            (
                choice.attrs.get("data-tooltip"),
                self.room.choose_switch_target,
                [choice.attrs.get("value")],
                {},
            )
            for choice in switch_targets
        ]

    def get_shifts(self):
        shift = self.soup.find_all(attrs={"name": "chooseShift"})
        return [
            (
                choice.attrs.get("data-tooltip"),
                self.room.choose_shift,
                [],
                {},
            )
            for choice in shift
        ]


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

        if any(prefix in data for prefix in {"|turn", "|teampreview"}):
            return True

        forceSwitch = self.room.get_js_attr("request?.forceSwitch")
        if "|request" not in data:
            return forceSwitch

    def reset(self):
        self.room.reset()
        self.choice = None
        self.request = None

    def get_state(self, raw: bool = False):
        self.state = self.room.get_state(raw=raw)
        return self.state

    def get_choices(self):
        return ChoiceBuilder(self.room).get_choices()
