import os
import re
import json

from extract import PS_CLIENT_DIR

CONDITIONS_PATH = f"pokemon-showdown-client/data/pokemon-showdown/data/conditions.ts"
MOVES_PATH = f"pokemon-showdown-client/data/pokemon-showdown/data/moves.ts"
BATTLE_PATH = f"pokemon-showdown-client/src/battle.ts"


def main():
    # os.system("npx prettier -w --tab-width 4 pokemon-showdown-client")

    src = ""
    with open(BATTLE_PATH, "r") as f:
        src += f.read()
    with open(CONDITIONS_PATH, "r") as f:
        src += f.read()
    with open(MOVES_PATH, "r") as f:
        src += f.read()

    for file in os.listdir(os.path.join(PS_CLIENT_DIR, "js")):
        if file.endswith(".js"):
            with open(
                os.path.join(PS_CLIENT_DIR, "js", file), "r", encoding="utf-8"
            ) as f:
                src += f.read()

    # with open("wsnc", "w", encoding="utf-8") as f:
    #     f.write(src)

    volatile_status = set()
    volatile_status.update(
        set(re.findall(r"removeVolatile\([\"|\'](.*?)[\"|\']\)", src))
    )
    volatile_status.update(set(re.findall(r"hasVolatile\([\"|\'](.*?)[\"|\']\)", src)))
    volatile_status.update(set(re.findall(r"volatiles\[[\"|\'](.*?)[\"|\']\]", src)))
    volatile_status.update(set(re.findall(r"volatiles\.(.*?)[\[|\)| ]", src)))
    volatile_status.update(
        set(re.findall(r"volatileStatus:\s*[\"|\'](.*)[\"|\'],", src))
    )
    volatile_status = list(volatile_status)
    for i, vs in enumerate(volatile_status):
        volatile_status[i] = "".join(c for c in vs if c.isalnum()).lower()
    volatile_status.sort()

    weathers = re.findall(r"[\"|\']-weather[\"|\'],\s*[\"|\'](.*)[\"|\'],", src)
    weathers = list(map(lambda s: s.lower(), sorted(set(weathers))))

    side_conditions = re.findall(r"sideCondition:\s*[\"|\'](.*)[\"|\'],", src)
    addSideCondition = re.search(
        r"addSideCondition\(.*\) \{([\S\s]*?)\n\t\}", src
    ).group()
    side_conditions += re.findall(r"case \"(.*?)\"", addSideCondition)
    side_conditions = list(sorted(set(side_conditions)))

    terrain = re.findall(r"terrain:\s*[\"|\'](.*)[\"|\'],", src)
    terrain = list(sorted(set(terrain)))

    pseudoweather = re.findall(r"pseudoWeather\:\s[\"|\'](.*?)[\"|\']", src)
    pseudoweather = list(sorted(set(pseudoweather)))

    item_effects = re.findall(r"itemEffect = [\"|\'](.*?)[\"|\']", src)
    item_effects = list(sorted(set(item_effects)))
    item_effects.pop(1)
    item_effects = set(item_effects)
    item_effects.update(["eaten", "popped", "consumed", "held up"])
    item_effects = list(sorted(item_effects))

    wsnc = {
        "weathers": weathers,
        "volatiles": volatile_status,
        "pseudoweather": pseudoweather,
        "terrain": terrain,
        "item_effects": item_effects,
        "side_conditions": side_conditions,
    }
    with open("meloetta/pretrained/wsnc.json", "w") as f:
        json.dump(wsnc, f)


if __name__ == "__main__":
    main()
