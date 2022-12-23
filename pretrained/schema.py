import re
import json
import numpy as np

from typing import List, Union

from tqdm import tqdm

from collections.abc import MutableMapping

from sklearn.preprocessing import OneHotEncoder

from meloetta.battle import Battle


SCHEMA = {
    "pokedex": {
        "id": str,
        "baseSpecies": str,
        "types": List[str],
        "baseStats.hp": float,
        "baseStats.atk": float,
        "baseStats.def": float,
        "baseStats.spa": float,
        "baseStats.spd": float,
        "baseStats.spa": float,
        "weightkg": float,
        "heightm": float,
    },
    "movedex": {
        {
            "accuracy",
            "basePower",
            "category",
            "critRatio",
            "flags.allyanim",
            "flags.bite",
            "flags.bullet",
            "flags.bypasssub",
            "flags.charge",
            "flags.contact",
            "flags.dance",
            "flags.distance",
            "flags.gravity",
            "flags.heal",
            "flags.mirror",
            "flags.nonsky",
            "flags.powder",
            "flags.protect",
            "flags.punch",
            "flags.reflectable",
            "flags.slicing",
            "flags.snatch",
            "flags.sound",
            "flags.wind",
            "hasCrashDamage",
            "heal",
            "isMax",
            "isZ",
            "maxMove.basePower",
            "multihit",
            "noPPBoosts",
            "noSketch",
            "ohko",
            "pp",
            "pressureTarget",
            "priority",
            "recoil",
            "secondaries",
            "target",
            "type",
            "zMove.basePower",
            "zMove.boost.accuracy",
            "zMove.boost.atk",
            "zMove.boost.def",
            "zMove.boost.evasion",
            "zMove.boost.spa",
            "zMove.boost.spd",
            "zMove.boost.spe",
            "zMove.effect",
        }
    },
    "itemdex": {
        "": int,
    },
    "abilitydex": {
        "": int,
    },
}


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


def get_pokedex(table, gen):
    try:
        pokedex = table[f"gen{gen}"]["overrideTier"]
    except:
        pokedex = table["overrideTier"]
    return sorted([p for p, s in pokedex.items() if s != "Illegal"])


def get_movedex(table, gen):
    try:
        movedex = table[f"gen{gen}"]["overrideMoveData"]
    except:
        movedex = table["overrideMoveData"]
    return sorted([p for p, s in movedex.items() if s.get("isNonstandard") != "Future"])


def get_itemdex(table, gen):
    try:
        itemdex = table[f"gen{gen}"]["items"]
    except:
        itemdex = table["items"]
    return sorted([i for i in itemdex if isinstance(i, str)])


def get_abilitydex(table, gen):
    try:
        abilitydex = table[f"gen{gen}"]["overrideAbilityData"]
    except:
        abilitydex = table["overrideAbilityData"]
    return sorted([p for p, s in abilitydex.items() if s != "Illegal"])


def main():

    format = "gen3randombattle"
    gen = int(re.search(r"gen([0-9])", format).groups()[0])

    battle = Battle()
    battle.set_gen(gen)

    with open("js/data/BattleTeambuilderTable.json", "r") as f:
        teambuilder_table = json.load(f)

    pokedex = get_pokedex(teambuilder_table, gen)
    movedex = get_movedex(teambuilder_table, gen)
    itemdex = get_itemdex(teambuilder_table, gen)
    abilitydex = get_abilitydex(teambuilder_table, gen)

    feature_vectors = {}
    for name, obj, func in [
        # ("pokedex", pokedex, battle.get_species),
        ("movedex", movedex, battle.get_move),
        ("itemdex", itemdex, battle.get_item),
        ("abilitydex", abilitydex, battle.get_ability),
    ]:

        feature_vectors[name] = {o: func(o) for o in obj}

        schema = {}
        for sample in feature_vectors[name].values():
            sample = flatten(sample)
            for key in list(sample):
                if key not in schema:  # and key in SCHEMA[name]:
                    schema[key] = set()

        schema.pop("rating", None)
        schema.pop("isNonstandard", None)
        schema.pop("desc", None)
        schema.pop("shortDesc", None)
        schema.pop("gen", None)
        schema.pop("megaStone", None)
        schema.pop("megaEvolves", None)

        ohes = {}

        for key in schema:
            for sample in feature_vectors[name].values():
                schema[key].add(str(get_nested(sample, key)))

            n = len(schema[key])
            if n <= 64:
                fit_ = np.array([sorted(list(schema[key]))])

                if key == "basePower":
                    u_values = np.unique((fit_.astype(float) ** 0.5).astype(int))
                    size = np.arange(u_values.max()).size + 1
                    ohes[key] = lambda x: np.expand_dims(np.eye(size)[int(x**0.5)], 0)

                elif key == "accuracy":

                    def acc_func(v):
                        if isinstance(v, bool):
                            v = np.array([[1, 0, 1]])
                        else:
                            v = np.array([[0, 1, v / 100]])

                        return v

                    ohes[key] = acc_func

                elif key == "pp":

                    def acc_func(v):
                        data = [[int(b) for b in f"{int(8 / 5 * v):#010b}"[2:]]]
                        return np.array(data)

                    ohes[key] = acc_func

                else:
                    ohes[key] = OneHotEncoder(max_categories=n)
                    ohes[key].fit(fit_.T)

        ohes = dict(sorted(ohes.items(), key=lambda x: x[0]))
        features = list(ohes.keys())
        for sample in tqdm(feature_vectors[name].values()):
            sample["ohes"] = {}
            for key, ohe in ohes.items():
                if key in ["basePower", "accuracy", "pp"]:
                    sample["ohes"][key] = ohe(sample[key])

                else:
                    val = str(get_nested(sample, key))
                    sample["ohes"][key] = ohe.transform([[val]]).toarray()

            feature_vector = np.concatenate(
                [sample["ohes"][feature] for feature in features], axis=-1
            )
            sample["feature_vector"] = feature_vector

        print()


if __name__ == "__main__":
    main()
