import os
import shutil

PS_CLIENT_DIR = "pokemon-showdown-client"

SRC = [
    f"{PS_CLIENT_DIR}/data/pokemon-showdown/.data-dist/abilities.js",
    f"{PS_CLIENT_DIR}/data/pokemon-showdown/.data-dist/aliases.js",
    f"{PS_CLIENT_DIR}/data/pokemon-showdown/.data-dist/items.js",
    f"{PS_CLIENT_DIR}/data/pokemon-showdown/.data-dist/moves.js",
    f"{PS_CLIENT_DIR}/data/pokemon-showdown/.data-dist/pokedex.js",
    f"{PS_CLIENT_DIR}/js/battle-scene-stub.js",
    f"{PS_CLIENT_DIR}/js/battle-choices.js",
    f"{PS_CLIENT_DIR}/js/battle-dex.js",
    f"{PS_CLIENT_DIR}/js/battle-dex-data.js",
    f"{PS_CLIENT_DIR}/js/battle-text-parser.js",
    f"{PS_CLIENT_DIR}/js/battle.js",
]


def main():
    for src_path in SRC:
        file = os.path.split(src_path)[-1]
        dst_path = os.path.join("js/client", file)
        shutil.copy(src_path, dst_path)


if __name__ == "__main__":
    main()
