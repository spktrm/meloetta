from typing import Union, Dict, Any, List
from py_mini_racer import MiniRacer

SRC = [
    "js/predef.js",
    "js/polyfill.js",
    "js/cycle.js",
    "js/client/abilities.js",
    "js/client/aliases.js",
    "js/client/items.js",
    "js/client/moves.js",
    "js/client/pokedex.js",
    "js/client/battle-scene-stub.js",
    "js/client/battle-choices.js",
    "js/client/battle-dex.js",
    "js/client/battle-dex-data.js",
    "js/client/battle-text-parser.js",
    "js/client/battle.js",
    "js/choices.js",
    "js/engine.js",
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


class Battle:
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
            self._ctx.eval(file_src)
        self._ctx.execute("engine.start()")

    def add(self, data: str = ""):
        self._ctx.call("engine.add", data)

    def instantAdd(self, data: str):
        self._ctx.call("engine.instandAdd", data)

    def push_to_step_queue(self, data: str):
        self._ctx.call("engine.addToStepQueue", data)

    def seek_turn(self, turn: int, force_reset: bool):
        self._ctx.call("engine.seekTurn", turn, force_reset)

    def setPerspective(self, sideid: str):
        self._ctx.call("engine.setPerspective", sideid)

    def parsePokemonId(self, pokemonid: str):
        return self._ctx.call("engine.parsePokemonId", pokemonid)

    def getPokemon(self, pokemonid: str):
        return self._ctx.call("engine.getPokemon", pokemonid)

    def fixRequest(self, request):
        return self._ctx.call("engine.fixRequest", request)

    def get_choices(self, request):
        return self._ctx.call("engine.getChoices", request)

    def reset(self):
        self._ctx.execute("engine.reset()")
        self._battle_tag = None
        self.request = None
        self.myPokemon = None
        self.ended = False

    def get_state(self, raw: bool = True) -> Dict[str, Union[str, Dict[str, Any]]]:
        state = self._ctx.call("engine.serialize")
        if getattr(self, "myPokemon") is not None:
            state["myPokemon"] = self.myPokemon
        if not raw:
            state = deserialize(state)
        return state

    @property
    def step_queue(self):
        return self.get_state().get("stepQueue", [])

    @property
    def action_required(self):
        return "|" == self.get_state().get("stepQueue", [""])[-1]

    @property
    def battle_tag(self):
        return self._battle_tag
