import re
import multiprocessing as mp

from typing import Tuple, List

from tqdm import tqdm

from meloetta.worker import SelfPlayWorker
from meloetta.buffers.buffer import ReplayBuffer
from meloetta.controllers.random import RandomController
from meloetta.controllers.naiveai import MewZeroController, Model


TRAJECTORY_LENGTH = 1024
NUM_BUFFERS = 1024

# BATTLE_FORMAT = "gen8randomdoublesbattle"
# BATTLE_FORMAT = "gen8randombattle"
BATTLE_FORMAT = "gen9randombattle"
# BATTLE_FORMAT = "gen8ou"
# BATTLE_FORMAT = "gen8doublesou"
# BATTLE_FORMAT = "gen3randombattle"
# BATTLE_FORMAT = "gen9doublesou"

TEAM = "null"
# TEAM = "Charizard||HeavyDutyBoots|Blaze|hurricane,fireblast,toxic,roost||85,,85,85,85,85||,0,,,,||88|]Venusaur||BlackSludge|Chlorophyll|leechseed,substitute,sleeppowder,sludgebomb||85,,85,85,85,85||,0,,,,||82|]Blastoise||WhiteHerb|Torrent|shellsmash,earthquake,icebeam,hydropump||85,85,85,85,85,85||||86|"
# TEAM = "Ceruledge||LifeOrb|WeakArmor|bitterblade,closecombat,shadowsneak,swordsdance||85,85,85,85,85,85||||82|,,,,,Fighting]Grafaiai||Leftovers|Prankster|encore,gunkshot,knockoff,partingshot||85,85,85,85,85,85||||86|,,,,,Dark]Greedent||SitrusBerry|CheekPouch|bodyslam,psychicfangs,swordsdance,firefang||85,85,85,85,85,85||||88|,,,,,Psychic]Quaquaval||LifeOrb|Moxie|aquastep,closecombat,swordsdance,icespinner||85,85,85,85,85,85||||80|,,,,,Fighting]Flapple||LifeOrb|Hustle|gravapple,outrage,dragondance,suckerpunch||85,85,85,85,85,85||||84|,,,,,Grass]Pachirisu||AssaultVest|VoltAbsorb|nuzzle,superfang,thunderbolt,uturn||85,85,85,85,85,85||||94|,,,,,Flying"


def get_gen_and_gametype(battle_format: str) -> Tuple[int, str]:
    gen = int(re.search(r"gen([0-9])", battle_format).groups()[0])
    if "triples" in battle_format:
        gametype = "triples"
    if "doubles" in battle_format:
        gametype = "doubles"
    else:
        gametype = "singles"
    return gen, gametype


def main():
    gen, gametype = get_gen_and_gametype(BATTLE_FORMAT)
    model = Model(gen=gen, gametype=gametype, embedding_dim=128)
    model.eval()

    replay_buffer = ReplayBuffer(TRAJECTORY_LENGTH, gen, gametype, NUM_BUFFERS)
    controller = MewZeroController(model, replay_buffer)

    procs: List[mp.Process] = []
    for i in range(16):  # This config will spawn 20 workers with 2 players each
        # for a total of 40 players, playing 20 games.
        # it is recommended to have an even number of players per worker

        worker = SelfPlayWorker(
            worker_index=i,
            num_players=2,  # 2 is players per worker
            battle_format=BATTLE_FORMAT,
            team=TEAM,
        )

        process = mp.Process(
            target=worker.run,
            args=(controller,),
            name=repr(worker),
        )
        procs.append(process)

    for proc in procs:
        proc.start()

    frame_monitor = tqdm(desc="frames: ")
    episode_monitor = tqdm(desc="eps: ")
    while True:
        _, frames = replay_buffer.finish_queue.get()
        frame_monitor.update(frames)
        episode_monitor.update(1)
        if not frames:
            break

    for proc in procs:
        proc.join()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
