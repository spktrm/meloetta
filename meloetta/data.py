import os
import json
import numpy as np

from copy import deepcopy
from typing import Any


DATA_DIR = "js/data"

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


BOOSTS = ["atk", "def", "spa", "spd", "spe", "evasion", "accuracy", "spc"]


def load_feature_embedding(type: str, gen: int):
    return np.load(os.path.join("pretrained", f"gen{gen}", type + ".npy"))


with open("pretrained/schema.json", "r") as f:
    schema = json.loads(f.read())

tokenized_schema = deepcopy(schema)

for gen in tokenized_schema:
    for dex_type in schema[gen]:
        for key, values in schema[gen][dex_type].items():
            tokenized_schema[gen][dex_type][key] = {
                str(values): index for index, values in enumerate(values)
            }


def get_species_token(gen: int, key: int, value: Any):
    lookup = tokenized_schema[f"gen{gen}"]["pokedex"][key]
    return lookup.get(value, -1)


def get_move_token(gen: int, key: int, value: Any):
    lookup = tokenized_schema[f"gen{gen}"]["movedex"][key]
    return lookup.get(value, -1)


def get_ability_token(gen: int, key: int, value: Any):
    lookup = tokenized_schema[f"gen{gen}"]["abilitydex"][key]
    return lookup.get(value, -1)


def get_item_token(gen: int, key: int, value: Any):
    lookup = tokenized_schema[f"gen{gen}"]["itemdex"][key]
    return lookup.get(value, -1)


GENDERS = {"M": 0, "F": 1, "N": 2}
STATUS = {"par": 0, "psn": 1, "frz": 2, "slp": 3, "brn": 4}


def get_gender_token(value: str):
    return GENDERS.get(value, -1)


def get_status_token(value: str):
    return STATUS.get(value, -1)


with open("pretrained/wsnc.json", "r") as f:
    WSNC = json.loads(f.read())

VOLATILES = WSNC["volatiles"]
WEATHERS = WSNC["weathers"]
PSEUDOWEATHER = WSNC["pseudoweather"]
TERRAIN = WSNC["terrain"]
ITEM_EFFECTS = WSNC["item_effects"]
