import re
import json
import torch

from typing import Dict, Union, Any, List

from meloetta.client import Client
from meloetta.room import BattleRoom
from meloetta.data import (
    get_choice_flag_token,
    get_choice_target_token,
    get_choice_token,
)
from meloetta.utils import expand_bt

from bs4 import BeautifulSoup


State = Dict[str, Union[str, Dict[str, Any], List[str]]]


class ChoiceBuilder:
    def __init__(self, room: BattleRoom):
        self.room = room
        tier = room.get_js_attr("battle.tier")
        self.gen = int(re.search(r"([0-9])", tier).group())
        self.gametype = room.get_js_attr("battle.gameType")
        controls = room.get_js_attr("controls.controls")

        self.choice = self.room.get_js_attr("choice")
        choices = self.choice.get("choices", [])
        choices = [c for c in choices if c is not None]
        self.choices: List[str] = choices
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
        action_masks = {}

        switch = False
        switch_mask = {index: False for index in range(6)}
        switches = self.get_switches() + self.get_teampreview()
        if switches:
            switch = True
            switch_mask.update(
                {int(action[0].split("|")[-1]): True for action in switches}
            )
        switch_mask = torch.tensor(list(switch_mask.values()))
        choices += switches

        move = False
        moves = self.get_moves()
        move_mask = {index: False for index in range(1, 5)}
        if moves:
            move = True
            move_mask.update({int(action[2][0]): True for action in moves})
        move_mask = torch.tensor(list(move_mask.values()))
        choices += moves

        targets = self.get_move_targets() + self.get_switch_targets()
        n = 2 if self.gametype == "doubles" else 3
        target_values = list(range(-n, n + 1))
        target_values.remove(0)
        target_mask = {index: False for index in target_values}
        if targets:
            targeting = torch.tensor(1)
            target_mask.update({int(action[2][0]): True for action in targets})
        else:
            targeting = torch.tensor(0)
        target_mask = torch.tensor(list(target_mask.values()))
        targeting = expand_bt(targeting)

        choices += targets

        tera = False
        max = False
        mega = False
        zmove = False

        if self.gen == 9:
            tera_moves = self.get_tera_moves()
            choices += tera_moves
            if tera_moves:
                tera = True

        if self.gen == 8:
            max_moves = self.get_max_moves()
            choices += max_moves
            if max_moves:
                max = True

        if self.gen == 7:
            zmoves = self.get_zmoves()
            choices += zmoves
            if zmoves:
                zmove = True

        if self.gen == 7 or self.gen == 6:
            mega_moves = self.get_mega()
            choices += mega_moves
            if mega_moves:
                mega = True

        if self.gametype == "triples" and self.pos != 1:
            choices += self.get_shifts()

        if self.gametype != "singles":  # only relevant for gamemodes not singles
            prev_choices = self.get_prev_choices()
        else:
            prev_choices = None

        action_type = torch.tensor([move, switch])
        flags = torch.tensor([mega, zmove, max, tera])

        action_masks["action_type"] = expand_bt(action_type)
        action_masks["valid_moves"] = expand_bt(move_mask)
        action_masks["valid_switches"] = expand_bt(switch_mask)
        action_masks["flags"] = expand_bt(flags)
        action_masks["targets"] = expand_bt(target_mask)

        return targeting, prev_choices, choices

    def get_prev_choices(self):
        choices = []
        total = 2 if self.gametype == "doubles" else 3
        remaining = total - len(self.choices) - 1
        for prev_choice in self.choices:
            prev_choice_token = -1
            index = -1
            flag_token = -1
            target_token = -1

            if prev_choice != "pass":
                token, index, *flags = prev_choice.split(" ")
                prev_choice_token = get_choice_token(token)
                index = int(index)

                if flags:
                    flag = ""
                    target = ""
                    for potential_flag in flags:
                        try:
                            int(potential_flag)
                        except ValueError:
                            flag = potential_flag
                        else:
                            target = potential_flag

                    if flag:
                        flag_token = get_choice_flag_token(flag)

                    if target:
                        target_token = get_choice_target_token(target)

            prev_choice_tensor = torch.tensor(
                [prev_choice_token, index, flag_token, target_token]
            )
            choices.append(prev_choice_tensor)

        done = len(self.choices) or self.choice.get("done", 0)
        choices += [torch.tensor([-1, -1, -1, -1]) for _ in range(remaining)]
        choices = expand_bt(torch.stack(choices))

        return {
            "choices": choices,
            "done": expand_bt(torch.tensor(done)),
        }

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
        mega_moves = []
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
        return mega_moves

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
        self.started = False
        self.request = None

    def get_state(self, raw: bool = False):
        return self.room.get_state(raw=raw)

    def get_choices(self):
        return ChoiceBuilder(self.room).get_choices()
