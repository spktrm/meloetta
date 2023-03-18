import re
import torch

from collections import OrderedDict
from typing import Union, NamedTuple, Tuple, Callable, Dict, Any, List

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


class Choices(NamedTuple):
    targeting: torch.Tensor
    prev_choices: torch.Tensor
    action_masks: Dict[str, torch.Tensor]
    choices: Dict[str, Tuple[Callable, List[Any], Dict[str, Any]]]


class ChoiceBuilder:
    def __init__(self, room: BattleRoom):
        self.room = room
        tier = room.get_js_attr("battle.tier")
        self.gen = int(re.search(r"([0-9])", tier).group())
        self.gametype = room.get_js_attr("battle.gameType")
        if self.gametype == "singles":
            self.n = 1
        if self.gametype == "doubles":
            self.n = 2
        if self.gametype == "triples":
            self.n = 3
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
        choices = {}
        action_masks = {}

        switch = False
        switch_mask = OrderedDict({index: False for index in range(6)})
        switches = self.get_switches()
        switches.update(self.get_teampreview())
        if switches:
            switch = True
            switch_mask.update({index: True for index in switches})

        switch_mask = torch.tensor(list(switch_mask.values()))
        choices["switches"] = switches

        move = False
        moves = self.get_moves()
        move_mask = OrderedDict({index: False for index in range(4)})
        max_move_mask = OrderedDict({index: False for index in range(4)})

        if moves:
            move = True
            move_mask.update({index: True for index in moves})

        move_targets = self.get_move_targets()
        switch_targets = self.get_switch_targets()
        targets = move_targets
        targets.update(switch_targets)

        target_values = list(range(2 * self.n))
        target_mask = OrderedDict({index: False for index in target_values})

        if targets:
            targeting = torch.tensor(1)
            target_mask.update({index: True for index in targets})
        else:
            targeting = torch.tensor(0)

        target_mask = torch.tensor(list(target_mask.values()))
        choices["targets"] = targets
        targeting = expand_bt(targeting)

        noflag = True
        tera = False
        max = False
        mega = False
        zmove = False

        if self.gen == 9:
            tera_moves = self.get_tera_moves()
            if tera_moves:
                choices["tera_moves"] = tera_moves
                move_mask.update({index: True for index in tera_moves})
                move = True
                tera = True

        if self.gen == 8:
            max_moves = self.get_max_moves()
            if max_moves:
                choices["max_moves"] = max_moves
                max_move_mask.update({index: True for index in max_moves})
                move = True
                max = True
                if not moves:
                    noflag = False

        if self.gen == 7:
            zmoves = self.get_zmoves()
            if zmoves:
                choices["zmoves"] = zmoves
                move_mask.update({index: True for index in zmoves})
                move = True
                zmove = True

        if self.gen == 7 or self.gen == 6:
            mega_moves = self.get_mega()
            if mega_moves:
                choices["mega_moves"] = mega_moves
                move_mask.update({index: True for index in mega_moves})
                move = True
                mega = True

        if self.gametype == "triples" and self.pos != 1:
            choices.update(self.get_shifts())

        if self.gametype != "singles":  # only relevant for gamemodes not singles
            prev_choices = self.get_prev_choices()
        else:
            prev_choices = {
                "prev_choices": None,
                "choices_done": None,
            }

        max_move_mask = torch.tensor(list(max_move_mask.values()))
        move_mask = torch.tensor(list(move_mask.values()))
        choices["moves"] = moves

        action_type = torch.tensor([move, switch, not (move or switch)])
        flags = torch.tensor([noflag, mega, zmove, max, tera])

        action_masks["action_type_mask"] = expand_bt(action_type)
        action_masks["move_mask"] = expand_bt(move_mask)
        if self.gen == 8:
            action_masks["max_move_mask"] = expand_bt(max_move_mask)
        else:
            action_masks["max_move_mask"] = None
        action_masks["switch_mask"] = expand_bt(switch_mask)
        action_masks["flag_mask"] = expand_bt(flags)
        if self.gametype != "singles" or self.gen == 9:
            action_masks["target_mask"] = expand_bt(target_mask)
        else:
            action_masks["target_mask"] = None

        return Choices(
            targeting,
            prev_choices,
            action_masks,
            choices,
        )

    def get_prev_choices(self):
        prev_choices = []
        total = 2 if self.gametype == "doubles" else 3
        remaining = total - len(self.choices)

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
            prev_choices.append(prev_choice_tensor)

        done = len(self.choices) or self.choice.get("done", 0)
        prev_choices += [torch.tensor([-1, -1, -1, -1]) for _ in range(remaining)]
        prev_choices = expand_bt(torch.stack(prev_choices))

        return {
            "prev_choices": prev_choices,
            "choices_done": expand_bt(torch.tensor(done)),
        }

    def get_teampreview(self):
        teampreview = self.soup.find_all(
            lambda tag: ("disabled" not in tag.attrs),
            attrs={"name": "chooseTeamPreview"},
        )
        choices = {}
        for choice in teampreview:
            index = choice.attrs.get("value")
            choices[int(index)] = (
                self.room.choose_team_preview,
                [choice.attrs.get("value")],
                {},
            )
        return choices

    def get_moves(self):
        self.reg_moves = self.soup.find_all(
            lambda tag: ("disabled" not in tag.attrs),
            attrs={
                "name": "chooseMove",
                "data-tooltip": re.compile(r"^move\|"),
            },
        )
        choices = {}
        for choice in self.reg_moves:
            index = choice.attrs.get("value")
            choices[int(index) - 1] = (
                self.room.choose_move,
                [index],
                {
                    "target": choice.attrs.get("data-target"),
                    "isMega": False,
                    "isZMove": False,
                    "isUltraBurst": False,
                    "isDynamax": False,
                    "isTerastal": False,
                },
            )
        return choices

    def get_max_moves(self):
        choices = {}
        if not self.isDynamax:  # and "dynamax" in self.checkboxes:
            max_moves = self.soup.find_all(
                lambda tag: ("disabled" not in tag.attrs),
                attrs={
                    "name": "chooseMove",
                    "data-tooltip": re.compile(r"^maxmove\|"),
                },
            )
            for choice in list(max_moves):
                index = choice.attrs.get("value")
                choices[int(index) - 1] = (
                    self.room.choose_move,
                    [index],
                    {
                        "target": choice.attrs.get("data-target"),
                        "isMega": False,
                        "isZMove": False,
                        "isUltraBurst": False,
                        "isDynamax": True,
                        "isTerastal": False,
                    },
                )
        return choices

    def get_mega(self):
        choices = {}
        if not self.isMega and "megaevo" in self.checkboxes:
            mega_moves = self.soup.find_all(
                lambda tag: ("disabled" not in tag.attrs),
                attrs={"name": "chooseMove"},
            )
            for choice in mega_moves:
                index = choice.attrs.get("value")
                choices[int(index) - 1] = (
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
        return choices

    def get_zmoves(self):
        choices = {}
        if not self.isZMove and "zmove" in self.checkboxes:
            z_moves = self.soup.find_all(
                lambda tag: ("disabled" not in tag.attrs),
                attrs={"name": "chooseMove", "data-tooltip": re.compile(r"^zmove\|")},
            )
            for choice in list(z_moves):
                index = choice.attrs.get("value")
                choices[int(index) - 1] = (
                    self.room.choose_move,
                    [index],
                    {
                        "target": choice.attrs.get("data-target"),
                        "isMega": False,
                        "isZMove": True,
                        "isUltraBurst": False,
                        "isDynamax": False,
                        "isTerastal": False,
                    },
                )
        return choices

    def get_tera_moves(self):
        choices = {}
        if not self.isTerastal and "terastallize" in self.checkboxes:
            for choice in self.reg_moves:
                index = choice.attrs.get("value")
                choices[int(index) - 1] = (
                    self.room.choose_move,
                    [index],
                    {
                        "target": choice.attrs.get("data-target"),
                        "isMega": False,
                        "isZMove": False,
                        "isUltraBurst": False,
                        "isDynamax": False,
                        "isTerastal": True,
                    },
                )
        return choices

    def get_move_targets(self):
        move_targets = self.soup.find_all(
            lambda tag: ("disabled" not in tag.attrs),
            attrs={"name": "chooseMoveTarget"},
        )
        choices = {}
        for choice in move_targets:
            index = choice.attrs.get("value")
            key = int(index)
            if key > 0:
                key -= 1
            key += self.n
            choices[key] = (self.room.choose_move_target, [index], {})
        return choices

    def get_switches(self):
        switches = self.soup.find_all(
            lambda tag: ("disabled" not in tag.attrs),
            attrs={"name": "chooseSwitch"},
        )
        choices = {}
        for choice in switches:
            index = choice.attrs.get("value")
            choices[int(index)] = (self.room.choose_switch, [index], {})
        return choices

    def get_switch_targets(self):
        switch_targets = self.soup.find_all(
            lambda tag: ("disabled" not in tag.attrs),
            attrs={"name": "chooseSwitchTarget"},
        )
        choices = {}
        for choice in switch_targets:
            index = choice.attrs.get("value")
            choices[int(index)] = (self.room.choose_switch_target, [index], {})
        return choices

    def get_shifts(self):
        shift = self.soup.find_all(
            lambda tag: ("disabled" not in tag.attrs),
            attrs={"name": "chooseShift"},
        )
        return {
            int(choice.attrs.get("value")): (self.room.choose_shift, [], {})
            for choice in shift
        }


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
        cls.started = False
        return cls

    async def recieve(self, data: str):
        return self._recieve(data)

    def _recieve(self, data: str):
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
