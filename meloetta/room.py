import re
import json

from typing import Union, Dict, Any, List
from py_mini_racer import MiniRacer

SRC = [
    "js/predef.js",
    "js/polyfill.js",
    "js/cycle.js",
    "js/underscore.js",
    "js/client/aliases.js",
    "js/client/abilities.js",
    "js/client/items.js",
    "js/client/moves.js",
    "js/client/pokedex.js",
    "js/client/typechart.js",
    "js/client/formats-data.js",
    "js/client/teambuilder-tables.js",
    "js/client/battle-scene-stub.js",
    "js/client/battle-choices.js",
    "js/client/battle-dex.js",
    "js/client/battle-dex-data.js",
    "js/client/battle-text-parser.js",
    "js/client/battle-tooltips.js",
    "js/client/battle-log.js",
    "js/client/battle.js",
    "js/client.js",
    "js/enginev2.js",
]


def deserialize(state: Dict[str, Union[str, List[str], Dict[str, Any]]]):
    def rez(value):
        nonlocal state
        if isinstance(value, list):
            for i, element in enumerate(value):
                if (
                    isinstance(element, dict) or isinstance(element, list)
                ) and element is not None:
                    try:
                        path = element.get("$ref")
                    except:
                        path = None
                    if isinstance(path, str):
                        path = path.replace("$", "state")
                        value[i] = eval(path)
                    else:
                        rez(element)
        elif isinstance(value, dict):
            for name in value.keys():
                item = value[name]
                if (
                    isinstance(item, dict) or isinstance(item, list)
                ) and item is not None:
                    try:
                        path = item.get("$ref")
                    except:
                        path = None
                    if isinstance(path, str):
                        path = path.replace("$", "state")
                        value[name] = eval(path)
                    else:
                        rez(item)

    rez(state)
    return state


class BattleRoom:
    title: str

    def __init__(self):
        self._ctx = MiniRacer()
        self._battle_tag = None
        self.myPokemon = None
        self.request = None
        self.ended = False
        self._load_js()

    def _load_js(self):
        for file in SRC:
            with open(file, "r", encoding="utf-8") as f:
                file_src = f.read()
            try:
                modules = re.findall(r"(module\.exports \= .*)", file_src)
                for module in modules:
                    file_src = file_src.replace(module, "")
                self._ctx.eval(file_src)
            except Exception as e:
                print(e)
                print(file)
                exit()
        self._ctx.eval("engine.start()")

    def recieve(self, data: str = ""):
        return self._ctx.execute("engine.receive({})".format(json.dumps(data)))

    def get_js_attr(self, attr: str):
        return self._ctx.execute(f"engine.client.{attr}")

    def get_battle(self, raw: bool = False):
        battle = self._ctx.execute("serialize(engine.client.battle)")
        if not raw:
            battle = deserialize(battle)
        return battle

    def add(self, data: str = ""):
        return self._call("engine.add", data)

    def _call(self, cmd, *args):
        js = cmd + "({})".format(json.dumps(args)[1:-1])
        return self._ctx.execute(js)

    def instantAdd(self, data: str):
        return self._call("engine.instandAdd", data)

    def push_to_step_queue(self, data: str):
        return self._call("engine.addToStepQueue", data)

    def seek_turn(self, turn: int, force_reset: bool):
        return self._call("engine.seekTurn", turn, force_reset)

    def setPerspective(self, sideid: str):
        return self._call("engine.setPerspective", sideid)

    def parsePokemonId(self, pokemonid: str):
        return self._call("engine.parsePokemonId", pokemonid)

    def getPokemon(self, pokemonid: str):
        return self._call("engine.getPokemon", pokemonid)

    def fixRequest(self, request):
        return self._call("engine.fixRequest", request)

    def get_choices(self, request):
        return self._call("engine.getChoices", request)

    def get_species(self, species):
        return self._call("engine.getSpecies", species)

    def get_move(self, move):
        return self._call("engine.getMove", move)

    def get_item(self, item):
        return self._call("engine.getItem", item)

    def get_ability(self, ability):
        return self._call("engine.getAbility", ability)

    def get_type(self, type):
        return self._call("engine.getType", type)

    def set_gen(self, gen):
        return self._call("engine.setGen", gen)

    def reset(self):
        self._ctx.eval("engine.reset()")
        self._battle_tag = None

    # choice start

    def choose_move_target(self, posString):
        return self._ctx.execute(
            "engine.chooseMoveTarget({})".format(json.dumps(posString))
        )

    def choose_move(
        self,
        pos: str,
        target: str,
        isMega: bool = False,
        isZMove: bool = False,
        isUltraBurst: bool = False,
        isDynamax: bool = False,
        isTerastal: bool = False,
    ):
        args = [pos, target, isMega, isZMove, isUltraBurst, isDynamax, isTerastal]
        args = json.dumps(args)[1:-1]
        cmd = "engine.chooseMove({})".format(args)
        return self._ctx.execute(cmd)

    def choose_shift(self):
        return self._ctx.execute("engine.chooseShift()")

    def choose_switch(self, pos: str):
        return self._ctx.execute("engine.chooseSwitch({})".format(json.dumps(pos)))

    def choose_switch_target(self, pos: str):
        return self._ctx.execute(
            "engine.chooseSwitchTarget({})".format(json.dumps(pos))
        )

    def choose_team_preview(self, pos: str):
        return self._ctx.execute("engine.chooseTeamPreview({})".format(json.dumps(pos)))

    def pop_outgoing(self):
        return self._ctx.execute("engine.popOutgoing()")

    # choice end

    def get_state(self, raw: bool = True) -> Dict[str, Union[str, Dict[str, Any]]]:
        state = self._ctx.execute("engine.serialize()")
        if not raw:
            state = deserialize(state)
        return state

    @property
    def battle_tag(self):
        return self._battle_tag


def main():
    room = BattleRoom()


if __name__ == "__main__":
    main()
