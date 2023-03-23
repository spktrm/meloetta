import os
import json
import numpy as np

from copy import deepcopy
from typing import Dict, Any


DATA_DIR = "meloetta/js/data"

with open(os.path.join(DATA_DIR, "BattleAbilities.json"), "r") as f:
    BattleAbilities = json.loads(f.read())

with open(os.path.join(DATA_DIR, "BattleAliases.json"), "r") as f:
    BattleAliases = json.loads(f.read())

with open(os.path.join(DATA_DIR, "BattleArticleTitles.json"), "r") as f:
    BattleArticleTitles = json.loads(f.read())

with open(os.path.join(DATA_DIR, "BattleFormatsData.json"), "r") as f:
    BattleFormatsData = json.loads(f.read())

with open(os.path.join(DATA_DIR, "BattleItems.json"), "r") as f:
    BattleItems = json.loads(f.read())

with open(os.path.join(DATA_DIR, "BattleLearnsets.json"), "r") as f:
    BattleLearnsets = json.loads(f.read())

with open(os.path.join(DATA_DIR, "BattleMovedex.json"), "r") as f:
    BattleMovedex = json.loads(f.read())

with open(os.path.join(DATA_DIR, "BattlePokedex.json"), "r") as f:
    BattlePokedex = json.loads(f.read())

with open(os.path.join(DATA_DIR, "BattleSearchCountIndex.json"), "r") as f:
    BattleSearchCountIndex = json.loads(f.read())

with open(os.path.join(DATA_DIR, "BattleSearchIndex.json"), "r") as f:
    BattleSearchIndex = json.loads(f.read())

with open(os.path.join(DATA_DIR, "BattleSearchIndexOffset.json"), "r") as f:
    BattleSearchIndexOffset = json.loads(f.read())

with open(os.path.join(DATA_DIR, "BattleTeambuilderTable.json"), "r") as f:
    BattleTeambuilderTable = json.loads(f.read())

with open(os.path.join(DATA_DIR, "BattleText.json"), "r") as f:
    BattleText = json.loads(f.read())

with open(os.path.join(DATA_DIR, "BattleTypeChart.json"), "r") as f:
    BattleTypeChart = json.loads(f.read())

with open(os.path.join(DATA_DIR, "Formats.json"), "r") as f:
    Formats = json.loads(f.read())


GMAX_MOVES = [move for move in BattleMovedex if "gmax" in move]


BOOSTS = ["atk", "def", "spa", "spd", "spe", "evasion", "accuracy", "spc"]


def load_feature_embedding(type: str, gen: int):
    return np.load(os.path.join("pretrained", f"gen{gen}", type + ".npy"))


Schema = Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]
with open("meloetta/pretrained/schema.json", "r") as f:
    schema: Schema = json.loads(f.read())

TOKENIZED_SCHEMA = deepcopy(schema)


def to_id(value: Any):
    if isinstance(value, str):
        return "".join([c for c in value if c.isalnum()]).lower()
    else:
        return str(value)


for gen in TOKENIZED_SCHEMA:
    for dex_type in schema[gen]:
        for key, values in sorted(schema[gen][dex_type].items()):
            TOKENIZED_SCHEMA[gen][dex_type][key] = {
                to_id(value): index
                for index, value in enumerate(values)
                if to_id(value)
            }


def get_type_token(gen: int, value: Any):
    value = to_id(value)
    lookup = TOKENIZED_SCHEMA[f"gen{gen}"]["movedex"]["type"]
    return lookup.get(value, -1)


def get_species_token(gen: int, key: int, value: Any):
    value = to_id(value)
    lookup = TOKENIZED_SCHEMA[f"gen{gen}"]["pokedex"][key]
    return lookup.get(value, -1)


def get_move_token(gen: int, key: int, value: Any):
    value = to_id(value)
    lookup = TOKENIZED_SCHEMA[f"gen{gen}"]["movedex"][key]
    return lookup.get(value, -1)


def get_ability_token(gen: int, key: int, value: Any):
    value = to_id(value)
    lookup = TOKENIZED_SCHEMA[f"gen{gen}"]["abilitydex"][key]
    return lookup.get(value, -1)


def get_item_token(gen: int, key: int, value: Any):
    value = to_id(value)
    lookup = TOKENIZED_SCHEMA[f"gen{gen}"]["itemdex"][key]
    return lookup.get(value, -1)


GENDERS = {"M": 0, "F": 1, "N": 2}
STATUS = {"par": 0, "psn": 1, "frz": 2, "slp": 3, "brn": 4}


def get_gender_token(value: str):
    return GENDERS.get(value, -1)


def get_status_token(value: str):
    return STATUS.get(value, -1)


with open("meloetta/pretrained/wsnc.json", "r") as f:
    WSNC = json.loads(f.read())

VOLATILES = WSNC["volatiles"]
VOLATILES = {v: i for i, v in enumerate(VOLATILES)}

WEATHERS = WSNC["weathers"]
WEATHERS = {v: i for i, v in enumerate(WEATHERS)}

PSEUDOWEATHERS = WSNC["pseudoweather"]
TERRAIN = WSNC["terrain"]
PSEUDOWEATHERS = {v: i for i, v in enumerate(PSEUDOWEATHERS + TERRAIN)}

ITEM_EFFECTS = WSNC["item_effects"]
ITEM_EFFECTS = {v: i for i, v in enumerate(ITEM_EFFECTS) if v}


SIDE_CONDITIONS = WSNC["side_conditions"]
SIDE_CONDITIONS = {v: i for i, v in enumerate(SIDE_CONDITIONS) if v}


def get_item_effect_token(name: str):
    return ITEM_EFFECTS.get(name, -1)


def get_weather_token(name: str):
    return WEATHERS.get(name, -1)


def get_pseudoweather_token(name: str):
    return PSEUDOWEATHERS.get(name, -1)


def get_side_condition_token(name: str):
    return SIDE_CONDITIONS.get(name, -1)


# Choice related

CHOICE_TOKENS = {
    "move": 0,
    "switch": 1,
}

CHOICE_FLAGS = {
    "mega": 0,
    "zmove": 1,
    "dynamax": 2,
    "max": 2,
    "terastallize": 3,
}

CHOICE_TARGETS = list(range(-3, 3))
CHOICE_TARGETS.remove(0)
CHOICE_TARGETS = {str(target): i for i, target in enumerate(CHOICE_TARGETS)}


def get_choice_flag_token(name: str):
    return CHOICE_FLAGS[name]


def get_choice_target_token(name: str):
    return CHOICE_TARGETS[name]


def get_choice_token(name: str):
    return CHOICE_TOKENS[name]


_STATE_FIELDS = {
    "sides",
    "boosts",
    "volatiles",
    "side_conditions",
    "pseudoweathers",
    "weather",
    "wisher",
    "turn",
    "n",
    "total_pokemon",
    "faint_counter",
    "turn",
    "prev_choices",
    "choices_done",
    "action_type_mask",
    "move_mask",
    "max_move_mask",
    "switch_mask",
    "flag_mask",
    "target_mask",
}
