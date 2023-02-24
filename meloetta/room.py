import re
import json

from typing import Union, Dict, Any, List
from py_mini_racer import MiniRacer
from py_mini_racer.py_mini_racer import is_unicode

SRC = [
    "meloetta/js/predef.js",
    "meloetta/js/polyfill.js",
    "meloetta/js/cycle.js",
    "meloetta/js/underscore.js",
    "meloetta/js/client/aliases.js",
    "meloetta/js/client/abilities.js",
    "meloetta/js/client/items.js",
    "meloetta/js/client/moves.js",
    "meloetta/js/client/pokedex.js",
    "meloetta/js/client/typechart.js",
    "meloetta/js/client/formats-data.js",
    "meloetta/js/client/teambuilder-tables.js",
    "meloetta/js/client/battle-scene-stub.js",
    "meloetta/js/client/battle-choices.js",
    "meloetta/js/client/battle-dex.js",
    "meloetta/js/client/battle-dex-data.js",
    "meloetta/js/client/battle-text-parser.js",
    "meloetta/js/client/battle-tooltips.js",
    "meloetta/js/client/battle-log.js",
    "meloetta/js/client/battle.js",
    "meloetta/js/client.js",
    "meloetta/js/engine.js",
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

    def _call(self, cmd, *args):
        js = cmd + "({})".format(json.dumps(args)[1:-1])
        return self._execute(js)

    def _execute(self, expr, timeout=None, max_memory=None):
        wrapped_expr = "JSON.stringify((function(){return (%s)})())" % expr
        ret = self._ctx.eval(wrapped_expr, timeout=timeout, max_memory=max_memory)
        if not is_unicode(ret):
            return None
        return self._ctx.json_impl.loads(ret)

    def recieve(self, data: str = ""):
        return self._execute("engine.receive({})".format(json.dumps(data)))

    def get_js_attr(self, attr: str):
        return self._execute(f"engine.client.{attr}")

    def get_battle(self, raw: bool = False):
        battle = self._execute("engine.serializeBattle()")
        if not raw:
            battle = deserialize(battle)
        return battle

    def get_species(self, species):
        return self._call("engine.getSpecies", species)

    def get_move(self, move):
        return self._call("engine.getMove", move)

    def get_item(self, item):
        return self._call("engine.getItem", item)

    def get_ability(self, ability):
        return self._call("engine.getAbility", ability)

    def set_gen(self, gen):
        return self._call("engine.setGen", gen)

    def get_pid(self):
        return self._execute("engine.getPid()")

    def get_reward(self):
        return self._execute("engine.getReward()")

    def reset(self):
        self._ctx.eval("engine.reset()")
        self._battle_tag = None

    # choice start

    def choose_move_target(self, posString):
        return self._execute(
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
        return self._execute(cmd)

    def choose_shift(self):
        return self._execute("engine.chooseShift()")

    def choose_switch(self, pos: str):
        return self._execute("engine.chooseSwitch({})".format(json.dumps(pos)))

    def choose_switch_target(self, pos: str):
        return self._execute("engine.chooseSwitchTarget({})".format(json.dumps(pos)))

    def choose_team_preview(self, pos: str):
        return self._execute("engine.chooseTeamPreview({})".format(json.dumps(pos)))

    def pop_outgoing(self):
        return self._execute("engine.popOutgoing()")

    # choice end

    def get_state(self, raw: bool = True) -> Dict[str, Union[str, Dict[str, Any]]]:
        state = self._execute("engine.serialize()")
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
