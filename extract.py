import re
import os
import json
import shutil
import platform

from py_mini_racer import MiniRacer


PS_CLIENT_DIR = "pokemon-showdown-client"

OS_NAME = "windows" if platform.system() == "Windows" else "unix"
DATA_DIR = {"windows": ".data-dist", "unix": "dist/data"}

CLIENT_SRC = [
    f"{PS_CLIENT_DIR}/data/pokemon-showdown/{DATA_DIR[OS_NAME]}/abilities.js",
    f"{PS_CLIENT_DIR}/data/pokemon-showdown/{DATA_DIR[OS_NAME]}/aliases.js",
    f"{PS_CLIENT_DIR}/data/pokemon-showdown/{DATA_DIR[OS_NAME]}/items.js",
    f"{PS_CLIENT_DIR}/data/pokemon-showdown/{DATA_DIR[OS_NAME]}/moves.js",
    f"{PS_CLIENT_DIR}/data/pokemon-showdown/{DATA_DIR[OS_NAME]}/pokedex.js",
    f"{PS_CLIENT_DIR}/data/pokemon-showdown/{DATA_DIR[OS_NAME]}/typechart.js",
    f"{PS_CLIENT_DIR}/data/pokemon-showdown/{DATA_DIR[OS_NAME]}/natures.js",
    f"{PS_CLIENT_DIR}/data/pokemon-showdown/{DATA_DIR[OS_NAME]}/conditions.js",
    f"{PS_CLIENT_DIR}/data/pokemon-showdown/{DATA_DIR[OS_NAME]}/learnsets.js",
    f"{PS_CLIENT_DIR}/data/pokemon-showdown/{DATA_DIR[OS_NAME]}/formats-data.js",
    f"{PS_CLIENT_DIR}/data/pokemon-showdown/{DATA_DIR[OS_NAME]}/tags.js",
    f"{PS_CLIENT_DIR}/data/teambuilder-tables.js",
    f"{PS_CLIENT_DIR}/js/battle-scene-stub.js",
    f"{PS_CLIENT_DIR}/js/battle-choices.js",
    f"{PS_CLIENT_DIR}/js/battle-dex.js",
    f"{PS_CLIENT_DIR}/js/battle-dex-data.js",
    f"{PS_CLIENT_DIR}/js/battle-text-parser.js",
    f"{PS_CLIENT_DIR}/js/battle.js",
]


DATA_SRC = [
    os.path.join(f"{PS_CLIENT_DIR}/data", file)
    for file in os.listdir(f"{PS_CLIENT_DIR}/data")
    if file.endswith(".js") and file not in ["text-afd.js"]
]


def main():
    for src_path in CLIENT_SRC:
        file = os.path.split(src_path)[-1]
        dst_path = os.path.join("js/client", file)
        shutil.copy(src_path, dst_path)

    _ctx = MiniRacer()
    exports = []
    for file in DATA_SRC:
        with open(file, "r", encoding="utf-8") as f:
            file_src = f.read()

        exports += re.findall(r"exports.(\w+) =", file_src)

        file_src = file_src.replace("exports.", "")
        _ctx.eval(file_src)

    for export in exports:
        try:
            string = _ctx.execute("JSON.stringify({})".format(export))
            obj = json.loads(string)
        except:
            print(f"{export} failed to write")
        else:
            with open(f"js/data/{export}.json", "w") as f:
                json.dump(obj, f)

    try:
        os.system("npx prettier -w --tab-width 4 js")
    except:
        print("[OPTIONAL] run `npx install prettier` to format javascript files after copying")


if __name__ == "__main__":
    main()
