import os
import json
import numpy as np

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


def load_feature_embedding(type: str, gen: int):
    return np.load(os.path.join("pretrained", f"gen{gen}", type + ".npy"))
