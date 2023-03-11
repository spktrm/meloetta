import os
import json
import torch
import traceback
import numpy as np

from copy import deepcopy
from abc import ABC
from tqdm import tqdm
from typing import Callable, List, Dict, Any, Set
from collections.abc import MutableMapping

from meloetta.data import GMAX_MOVES
from meloetta.room import BattleRoom


from transformers import AutoTokenizer, AutoModel

# model_id = "princeton-nlp/sup-simcse-roberta-large"
model_id = "dmis-lab/biobert-v1.1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)
model = model.eval()
# model = model.to(device)


def bin_enc(v, n):
    code = f"0{n+2}b"
    data = [int(b) for b in format(int(v), code)[2:]]
    return np.array(data)


@torch.no_grad()
def vectorize_text(text):
    assert text is not None and text
    encoded = tokenizer([text], padding=True, truncation=True, return_tensors="pt")
    vector = model(**encoded).pooler_output.squeeze()
    return vector.numpy()


def one_hot_enc(v, n):
    return np.eye(n)[v]


def sqrt_one_hot_enc(v, n):
    return np.eye(n)[v]


def bow_enc(vs, n):
    m = np.eye(n)
    return np.stack([m[i] for i in vs]).sum(0)


def secondary_enc(chances, effects, num):
    enc = np.zeros(num)
    for chance, effect in zip(chances, effects):
        enc[effect] = chance
    return enc


def single_enc(value):
    return np.array([1 if value is not None else 0])


def z_score(value, mu, sigma):
    return np.array([(value - mu) / sigma])


def flatten(d):
    items = []
    for k, v in d.items():
        if isinstance(v, MutableMapping):
            items.extend([(f"{k}.{t1}", t2) for t1, t2 in flatten(v).items()])
        else:
            items.append((k, v))
    return dict(items)


def get_nested(d, field):
    subfields = field.split(".")
    v = d
    for sf in subfields:
        v = v.get(sf, {})
        if v is None:
            return v
        if not isinstance(v, dict) and not isinstance(v, np.ndarray):
            return v


def get_schema(data: dict):
    schema: Dict[str, set] = {}
    for sample in data.values():
        sample = flatten(sample)
        for key in list(sample):
            if key not in schema:
                schema[key] = set()

    for key in schema:
        for sample in data.values():
            schema[key].add(json.dumps(get_nested(sample, key)))

    schema = dict(sorted(schema.items()))
    return {key: sorted(value) for key, value in schema.items() if len(value) > 1}


class Dex(ABC):
    FEATURES: Dict[str, Callable]
    schema: Dict[str, List[Any]]
    data: Dict[str, List[Any]]

    def stat_statistics(self, stat: str):
        stats = {}
        raw = [p[stat] for p in self.data.values()]
        stats["mu"] = np.mean(raw)
        stats["sigma"] = np.std(raw)
        return stats

    def vectorize(self, feature: str, value: Any):
        func = self.FEATURES.get(feature)
        schema = self.schema[feature]
        if func.__name__ == "one_hot_enc":
            num = len(schema)
            value = schema.index(json.dumps(value))
            return func(value, num)
        elif "bin_enc" in func.__name__:
            value = value or 0
            num = max(list(map(lambda x: int(float(x)) if x != "null" else 0, schema)))
            num = int(np.ceil(np.log2(num)))
            return func(value, num)
        elif "log_enc" in func.__name__:
            return func(value)
        elif func.__name__ == "sqrt_one_hot_enc":
            num = max(list(map(lambda x: int(x), schema)))
            num = int(np.sqrt(num))
            value = int(np.sqrt(value))
            return func(value, num)
        elif func.__name__ == "bow_enc":
            cats = set()
            for p in schema:
                p = json.loads(p)
                cats.update(p)
            cats = list(sorted(cats))
            num = len(cats)
            return func([cats.index(v) for v in value], num)
        elif func.__name__ == "secondary_enc":
            cats = set()
            for p in schema:
                p = json.loads(p)
                if p is not None:
                    for eff in p:
                        eff = json.dumps(
                            {k: v for k, v in eff.items() if k != "chance"}
                        )
                        cats.add(eff)
            cats = list(sorted(cats))
            num = len(cats)
            value = value or []
            chances = [v.pop("chance", 0) / 100 for v in value]
            effects = [cats.index(json.dumps(v)) for v in value]
            return func(chances, effects, num)
        elif func.__name__ == "z_score":
            splits = feature.split(".")
            if len(splits) > 1:
                _, feature = splits
                mu_sigma = getattr(self, f"base_stat_mu_sigma")[feature]
            else:
                mu_sigma = getattr(self, f"{feature}_mu_sigma")
            norm_value = func(value, mu_sigma["mu"], mu_sigma["sigma"])
            return norm_value
        else:
            return func(value)


class Pokedex(Dex):
    FEATURES = {
        # "id": one_hot_enc,
        # "baseSpecies": one_hot_enc,
        "formeid": one_hot_enc,
        "types": bow_enc,
        "abilities.S": one_hot_enc,
        "baseStats.hp": z_score,
        "baseStats.atk": z_score,
        "baseStats.def": z_score,
        "baseStats.spa": z_score,
        "baseStats.spd": z_score,
        "baseStats.spe": z_score,
        "bst": z_score,
        "weightkg": z_score,
        # "heightm": bin_enc,
        "evos": lambda v: one_hot_enc(int(True if v else False), 2),
    }

    def __init__(self, battle: BattleRoom, table: Dict[str, Any], gen: int):
        self.dex = self.get_dex(table, gen)
        data = {o: battle.get_species(o) for o in self.dex}
        self.data = {k: v for k, v in data.items() if v.get("exists", False)}
        self.raw_schema = get_schema(data)
        self.schema = {k: v for k, v in self.raw_schema.items() if k in self.FEATURES}

        self.base_stat_mu_sigma = self.base_stat_statistics()
        self.heightm_mu_sigma = self.stat_statistics("heightm")
        self.weightkg_mu_sigma = self.stat_statistics("weightkg")
        self.bst_mu_sigma = self.stat_statistics("bst")

        self.abilities = set()
        for d in data.values():
            abilities = set(d.get("abilities", {}).values())
            self.abilities.update(abilities)

        for ability, d in (
            table.get(f"gen{gen}", {}).get("overrideAbilityData", {}).items()
        ):
            if "isNonstandard" in d and d["isNonstandard"] is None:
                self.abilities.add(ability)

        self.moves = set()
        for name in self.dex:
            if name in table["learnsets"]:
                self.moves.update(
                    [
                        move
                        for move, gens in table["learnsets"][name].items()
                        if str(gen) in gens
                    ]
                )

        for move, d in table.get(f"gen{gen}", {}).get("overrideMoveData", {}).items():
            if "isNonstandard" in d and d["isNonstandard"] is None:
                self.moves.add(move)

    def base_stat_statistics(self):
        stats = {}
        for stat in {"hp", "atk", "def", "spa", "spd", "spe"}:
            stats[stat] = {}
            try:
                raw = [p["baseStats"][stat] for p in self.data.values()]
                stats[stat]["mu"] = np.mean(raw)
                stats[stat]["sigma"] = np.std(raw)
            except:
                pass
        return stats

    def get_dex(self, table, gen):
        try:
            pokedex = table[f"gen{gen}"]["overrideTier"]
        except:
            pokedex = table["overrideTier"]
        return sorted(
            [p for p, s in pokedex.items() if s != "Illegal" and "CAP" not in s]
        )


class Movedex(Dex):
    FEATURES = {
        "accuracy": lambda x: np.array([0, 1, int(x) / 100])
        if x == True
        else np.array([1, 0, 1]),
        "basePower": z_score,
        "category": one_hot_enc,
        "critRatio": one_hot_enc,
        "flags.allyanim": single_enc,
        "flags.bite": single_enc,
        "flags.bullet": single_enc,
        "flags.bypasssub": single_enc,
        "flags.charge": single_enc,
        "flags.contact": single_enc,
        "flags.dance": single_enc,
        "flags.defrost": single_enc,
        "flags.distance": single_enc,
        "flags.gravity": single_enc,
        "flags.heal": single_enc,
        "flags.mirror": single_enc,
        "flags.nonsky": single_enc,
        "flags.powder": single_enc,
        "flags.protect": single_enc,
        "flags.pulse": single_enc,
        "flags.punch": single_enc,
        "flags.recharge": single_enc,
        "flags.reflectable": single_enc,
        "flags.slicing": single_enc,
        "flags.snatch": single_enc,
        "flags.sound": single_enc,
        "flags.wind": single_enc,
        "hasCrashDamage": one_hot_enc,
        "heal": one_hot_enc,
        # "id": one_hot_enc,
        # "maxMove.basePower": bin_enc,
        "multihit": one_hot_enc,
        "noPPBoosts": one_hot_enc,
        "noSketch": one_hot_enc,
        # "num": one_hot_enc,
        "ohko": one_hot_enc,
        "pp": bin_enc,
        "pressureTarget": one_hot_enc,
        "priority": one_hot_enc,
        "recoil": one_hot_enc,
        # "secondaries": secondary_enc,
        "target": one_hot_enc,
        "type": one_hot_enc,
        "desc": vectorize_text,
        # "zMove.basePower": bin_enc,
        # "zMove.boost.accuracy": one_hot_enc,
        # "zMove.boost.atk": one_hot_enc,
        # "zMove.boost.def": one_hot_enc,
        # "zMove.boost.evasion": one_hot_enc,
        # "zMove.boost.spa": one_hot_enc,
        # "zMove.boost.spd": one_hot_enc,
        # "zMove.boost.spe": one_hot_enc,
        # "zMove.effect": one_hot_enc,
    }

    def __init__(self, battle: BattleRoom, moves: Set[str], gen: int):
        if gen == 8:
            moves.update(GMAX_MOVES)
        data = {move: battle.get_move(move) for move in moves}

        with open("meloetta/js/data/BattleMovedex.json") as f:
            self.raw_json = json.load(f)

        for k in data:
            data[k]["shortDesc"] = self.raw_json[k]["shortDesc"]
            data[k]["desc"] = self.raw_json[k]["desc"]

        self.data = {k: v for k, v in data.items() if v.get("exists", False)}

        self.raw_schema = get_schema(data)
        self.schema = {
            k: self.raw_schema[k] for k in self.FEATURES if k in self.raw_schema
        }

        self.basePower_mu_sigma = self.stat_statistics("basePower")

        keys_to_pop = []
        if gen < 7:
            for key in self.schema:
                if "zMove" in key:
                    keys_to_pop.append(key)
        if gen < 8:
            for key in self.schema:
                if "maxMove" in key:
                    keys_to_pop.append(key)

        for key in keys_to_pop:
            self.schema.pop(key)


class Itemdex(Dex):
    FEATURES = {
        "fling.basePower": one_hot_enc,
        "fling.status": one_hot_enc,
        "fling.volatileStatus": one_hot_enc,
        # "id": one_hot_enc,
        "isPokeball": one_hot_enc,
        "naturalGift.basePower": bin_enc,
        "naturalGift.type": one_hot_enc,
        "onPlate": one_hot_enc,
        "desc": vectorize_text,
    }

    def __init__(self, battle: BattleRoom, table: Dict[str, Any], gen: int):
        self.dex = self.get_dex(table, gen)
        data = {o: battle.get_item(o) for o in self.dex}

        with open("meloetta/js/data/BattleItems.json") as f:
            self.raw_json = json.load(f)

        for k in data:
            data[k]["shortDesc"] = self.raw_json.get(k, {}).get("shortDesc", "")
            data[k]["desc"] = self.raw_json.get(k, {}).get("desc", "")

        self.data = {k: v for k, v in data.items() if v.get("exists", False)}
        self.raw_schema = get_schema(data)
        self.schema = {k: v for k, v in self.raw_schema.items() if k in self.FEATURES}

    def get_dex(self, table, gen):
        try:
            itemdex = table[f"gen{gen}"]["items"]
        except:
            itemdex = table["items"]
        return sorted([i for i in itemdex if isinstance(i, str)])


class Abilitydex(Dex):
    FEATURES = {
        # "id": one_hot_enc,
        "isPermanent": one_hot_enc,
        "desc": vectorize_text,
    }

    def __init__(self, battle: BattleRoom, abilities: List[str], gen: int):
        data = {ability: battle.get_ability(ability) for ability in abilities}

        with open("meloetta/js/data/BattleAbilities.json") as f:
            self.raw_json = json.load(f)

        for k, v in data.items():
            data[k]["shortDesc"] = self.raw_json.get(v["id"], {}).get("shortDesc", "")
            data[k]["desc"] = self.raw_json.get(v["id"], {}).get("desc", "")

        self.data = {k: v for k, v in data.items() if v.get("exists", False)}
        self.raw_schema = get_schema(data)
        self.schema = {k: v for k, v in self.raw_schema.items() if k in self.FEATURES}


def main():
    # os.system("npx prettier -w --tab-width 4 pokemon-showdown-client")

    schema = {}
    for gen in range(8, 10):
        battle = BattleRoom()
        battle.set_gen(gen)

        with open("meloetta/js/data/BattleTeambuilderTable.json", "r") as f:
            teambuilder_table = json.load(f)

        save_dir = f"meloetta/pretrained/gen{gen}"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        pokedex = Pokedex(battle, teambuilder_table, gen)
        movedex = Movedex(battle, pokedex.moves, gen)
        abilitydex = Abilitydex(battle, pokedex.abilities, gen)
        itemdex = Itemdex(battle, teambuilder_table, gen)

        schema[f"gen{gen}"] = {}

        for dex in [
            pokedex,
            movedex,
            itemdex,
            abilitydex,
        ]:
            dex: Dex

            dex_name = type(dex).__name__.lower()
            samples = sorted(dex.data.values(), key=lambda x: x["name"])
            progress = tqdm(samples, desc=f"gen{gen}: " + dex_name)
            try:
                for sample in progress:
                    # progress.set_description(sample.get("name", ""))
                    sample["encs"] = {}

                    for feature in dex.schema:
                        value = get_nested(sample, feature)
                        if feature == "desc":
                            if value is None:
                                value = (
                                    value
                                    or dex.raw_json[sample["name"]].get("shortDesc")
                                    or get_nested(sample, "desc")
                                    or dex.raw_json[sample["name"]].get("desc")
                                )
                        sample["encs"][feature] = dex.vectorize(feature, value)

                    feature_vector = np.concatenate(
                        [sample["encs"][feature] for feature in dex.schema]
                    )
                    sample["feature_vector"] = torch.from_numpy(feature_vector)
                data = torch.stack([sample["feature_vector"] for sample in samples])
                data = torch.cat((torch.zeros_like(data[0]).unsqueeze(0), data), dim=0)
                names = [None] + [sample["name"] for sample in samples]
            except:
                traceback.print_exc()
            else:
                schema[f"gen{gen}"][dex_name] = deepcopy(dex.raw_schema)
                for key, values in schema[f"gen{gen}"][dex_name].items():
                    for index, value in enumerate(values):
                        schema[f"gen{gen}"][dex_name][key][index] = json.loads(value)
                save_path = os.path.join(save_dir, dex_name + ".pt")

                print(f"data.shape: {data.shape}")
                torch.save((names, data), save_path)

    with open("meloetta/pretrained/schema.json", "w") as f:
        json.dump(schema, f)


if __name__ == "__main__":
    main()
