import os
import re
import json

CLIENT_SRC = "pokemon-showdown-client/js"


CONDITIONS_PATH = "pokemon-showdown-client/data/pokemon-showdown/data/conditions.ts"
MOVES_PATH = "pokemon-showdown-client/data/pokemon-showdown/data/moves.ts"


def main():
    src = ""
    with open(CONDITIONS_PATH, "r") as f:
        src += f.read()
    with open(MOVES_PATH, "r") as f:
        src += f.read()

    for file in os.listdir(CLIENT_SRC):
        if file.endswith(".js"):
            with open(os.path.join(CLIENT_SRC, file), "r") as f:
                src += f.read()

    volatile_status = re.findall(r"removeVolatile\(\"(.*?)\"\)", src)
    volatile_status += re.findall(r"hasVolatile\(\"(.*?)\"\)", src)
    volatile_status += re.findall(r"volatiles\[\"(.*?)\"\]", src)
    volatile_status += re.findall(r"volatiles\.(.*?)[\[|\)| ]", src)
    volatile_status += re.findall(r"volatileStatus:\s*\"(.*)\",", src)
    for i, vs in enumerate(volatile_status):
        volatile_status[i] = "".join(c for c in vs if c.isalnum()).lower()
    volatile_status = list(sorted(set(volatile_status)))

    weathers = re.findall(r"\"-weather\",\s*\"(.*)\",", src)
    weathers = list(sorted(set(weathers)))

    side_conditions = re.findall(r"sideCondition:\s*\"(.*)\",", src)
    side_conditions = list(sorted(set(side_conditions)))

    terrain = re.findall(r"terrain:\s*\"(.*)\",", src)
    terrain = list(sorted(set(terrain)))

    pseudo_weather = re.findall(r"pseudoWeather\: \"(.*?)\"", src)
    pseudo_weather = list(sorted(set(pseudo_weather)))

    item_effects = re.findall(r"itemEffect \= \"(.*?)\"", src)
    item_effects = list(sorted(set(item_effects)))
    item_effects.pop(1)
    new_item_effects = []
    for item_effect in item_effects[1:]:
        new_item_effects.append(f"({item_effect})")
    item_effects += new_item_effects

    print()


if __name__ == "__main__":
    main()
