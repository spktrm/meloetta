import re
import json
import traceback

from typing import Dict, Union, Any, List

from meloetta.client import Client
from meloetta.battle import Battle
from meloetta.state import VectorizedState


class ChoiceBuilder:
    def __init__(
        self, battle, request: Dict[str, Union[str, Dict[str, Any], List[str]]]
    ):
        self.battle = battle
        self.request = request
        self.choice_type = request.get("requestType")

        # for moves
        self.move = 0
        self.target = 0
        self.mega = False
        self.ultra = False
        self.z = False
        self.max = False
        self.tera = False

        # for switches
        self.target = 0

        self.build_choices()

    def build_choices(self):
        if self.choice_type == "move":
            self.updateMoveControls()
        elif self.choice_type == "switch":
            self.updateSwitchControls()
        elif self.choice_type == "team":
            self.updateTeamControls()
        else:
            self.updateWaitControls()

    def updateMoveControls(self):
        return

    def updateSwitchControls(self):
        return

    def updateTeamControls(self):
        return

    def updateWaitControls(self):
        return

    def possible_moves(self):
        return


class Player:
    client: Client
    battle: Battle
    choice: str
    action_required: bool
    request: Dict[str, Union[str, Dict[str, Any]]]

    @classmethod
    async def create(cls, username, password, address) -> "Player":
        cls = Player()
        cls.client = await Client.create(username, password, address)
        cls.battle = Battle()
        cls.action_required = False
        cls.request = None
        return cls

    async def recieve(self, data: str):
        return self.add(data)

    def init(self, data: str):
        log = data.split("\n")
        if data[:6] == "|init|":
            log.pop(0)
        if len(log) and log[0][:7] == "|title|":
            self.battle.title = log[0][7:]
            log.pop(0)

        if len(self.battle.step_queue):
            return

        self.battle.step_queue = log
        self.battle.seek_turn(float("inf"), True)
        if self.battle.ended:
            self.battleEnded = True

    def receiveRequest(
        self, request: Dict[str, Union[Dict[str, Any], str]], choiceText
    ):
        if not request:
            self.side = ""
            return

        request["requestType"] = "move"
        if request.get("forceSwitch"):
            request["requestType"] = "switch"
        elif request.get("teamPreview"):
            request["requestType"] = "team"
        elif request.get("wait"):
            request["requestType"] = "wait"

        request = self.battle.fixRequest(request)

        self.choice = {"waiting": True} if choiceText else None
        self.request = request
        self.battle.request = self.request

        if request.get("side"):
            self.battle.myPokemon = request["side"]["pokemon"]
            self.battle.setPerspective(request["side"]["id"])
            self.updateSideLocation(request["side"])

    def updateSideLocation(self, sideData: Dict[str, Any]):
        if not sideData.get("id"):
            return
        self.side = sideData["id"]
        state = self.battle.get_state()
        if state["mySide"]["sideid"] != self.side:
            self.battle.setPerspective(self.side)

    def add(self, data: str):
        action_required = False
        if data.startswith(">"):
            try:
                nlIndex = data.index("\n")
            except IndexError:
                nlIndex = -1
            self.battle._battle_tag = data[1:nlIndex]
            data = data[nlIndex + 1 :]
        if not data:
            return
        if data[:6] == "|init|":
            return self.init(data)
        if data[:9] == "|request|":
            data = data[9:]

            requestData = None
            choiceText = None

            try:
                nlIndex = data.index("\n")
            except:
                nlIndex = -1

            try:
                c1 = re.search(r"[0-9]", data[0])
            except IndexError:
                c1 = None

            try:
                c2 = data[1] == "|"
            except IndexError:
                c2 = None

            if c1 and c2:
                # message format:
                #   |request|CHOICEINDEX|CHOICEDATA
                #   REQUEST

                # This is backwards compatibility with old code that violates the
                # expectation that server messages can be streamed line-by-line.
                # Please do NOT EVER push protocol changes without a pull request.
                # https://github.com/Zarel/Pokemon-Showdown/commit/e3c6cbe4b91740f3edc8c31a1158b506f5786d72#commitcomment-21278523
                choiceText = "?"
                data = data[2:nlIndex]
            elif nlIndex >= 0:
                # message format:
                #   |request|REQUEST
                #   |sentchoice|CHOICE
                if data[nlIndex + 1 : nlIndex + 13] == "|sentchoice|":
                    choiceText = data[nlIndex + 13 :]

                data = data[:nlIndex]

            if data:
                try:
                    requestData = json.loads(data)
                except json.JSONDecodeError as e:
                    traceback.print_exc()
                return self.receiveRequest(requestData, choiceText)

        log = data.split("\n")
        for i in range(len(log)):
            logLine = log[i]
            if logLine.startswith("|turn"):
                action_required = True

            if logLine[:10] == "|callback|":
                # TODO: Maybe a more sophisticated UI for this.
                # In singles, this isn't really necessary because some elements of the UI will be
                # immediately disabled. However, in doubles/triples it might not be obvious why
                # the player is being asked to make a new decision without the following messages.
                args = logLine[10:].split("|")
                try:
                    pokemon = self.battle.getPokemon(args[1])
                except IndexError:
                    pokemon = self.battle.nearSide["active"][args[1]]
                requestData = (
                    self.request["active"][pokemon["slot"] if pokemon else 0]
                    if self.request.get("active")
                    else None
                )
                self.choice = None

                if args[0] == "trapped":
                    requestData["trapped"] = True

                if args[0] == "cant":
                    for i in range(len(requestData.get("moves", []))):
                        if requestData["moves"][i]["id"] == args[3]:
                            requestData["moves"][i]["disabled"] = True

                    slots = ["a", "b", "c", "d", "e", "f"]
                    ident = (
                        pokemon["ident"][:2]
                        + slots[pokemon["slot"]]
                        + pokemon["ident"][2:]
                    )
                    args[1] = ident
                    self.battle.push_to_step_queue("|" + "|".join(args))

            elif logLine[:7] == "|title|":
                # eslint-disable-line no-empty
                pass
            elif logLine[:5] == "|win|" or logLine == "|tie":
                self.battle.ended = True
                self.battle.push_to_step_queue(logLine)
            elif (
                logLine[:6] == "|chat|"
                or logLine[:3] == "|c|"
                or logLine[:4] == "|c:|"
                or logLine[:9] == "|chatmsg|"
                or logLine[:10] == "|inactive|"
            ):
                self.battle.instantAdd(logLine)
            else:
                self.battle.push_to_step_queue(logLine)

        self.battle.add()
        if not action_required:
            action_required = (self.request or {}).get("forceSwitch", False)
        return action_required

    def reset(self):
        self.battle.reset()
        self.choice = None
        self.request = None

    def get_state(self, raw: bool = False):
        self.state = self.battle.get_state(raw=raw)
        return self.state

    def get_vectorized_state(self):
        state = self.battle.get_state(raw=False)
        return VectorizedState(self, state)

    def get_choices(self):
        nchoices = self.state["pokemonControlled"]
        choices_ = self.battle.get_choices(self.request)
        possibilities = [[] for _ in range(nchoices)]
        trapped = False
        if self.request.get("active"):
            for ai, active in enumerate(self.request["active"]):
                possibilities[ai] += [
                    f"move {i + 1}"
                    for i, move in enumerate(active["moves"])
                    if move.get("pp", float("inf")) > 0 and not move.get("disabled")
                ]
                if active.get("canMegaEvo"):
                    possibilities[ai] += [
                        f"move {i + 1} mega"
                        for i, move in enumerate(active["moves"])
                        if move.get("pp", 0) > 0 and not move.get("disabled")
                    ]
                if active.get("canZ"):
                    possibilities[ai] += [
                        f"move {i + 1} mega"
                        for i, move in enumerate(active["moves"])
                        if move.get("pp", 0) > 0 and not move.get("disabled")
                    ]
                if active.get("canDynamax"):
                    possibilities[ai] += [
                        f"move {i + 1} max"
                        for i, move in enumerate(active["moves"])
                        if move.get("pp", 0) > 0 and not move.get("disabled")
                    ]
                if self.request.get("side") and not active.get("trapped", False):
                    pokemon = self.request["side"]["pokemon"]
                    possibilities[ai] += [
                        f"switch {i + 1}"
                        for i, p in enumerate(pokemon)
                        if not p.get("active", False) and p.get("condition") != "0 fnt"
                    ]
        else:
            for ai in range(nchoices):
                if self.request.get("side"):
                    pokemon = self.request["side"]["pokemon"]
                    possibilities[ai] += [
                        f"switch {i + 1}"
                        for i, p in enumerate(pokemon)
                        if not p.get("active", False) and p.get("condition") != "0 fnt"
                    ]
        return possibilities[0]

    @property
    def rqid(self) -> Union[str, None]:
        rqid = self.request.get("rqid")
        return str(rqid)
