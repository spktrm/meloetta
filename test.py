import multiprocessing as mp

from meloetta.worker import SelfPlayWorker
from meloetta.controllers.random import RandomController
from meloetta.controllers.naiveai import NaiveAIController


BATTLE_FORMAT = "gen8randomdoublesbattle"
# BATTLE_FORMAT = "gen3randombattle"
# BATTLE_FORMAT = "gen3randombattle"
# BATTLE_FORMAT = "gen9doublesou"

TEAM = "null"
# TEAM = "Charizard||HeavyDutyBoots|Blaze|hurricane,fireblast,toxic,roost||85,,85,85,85,85||,0,,,,||88|]Venusaur||BlackSludge|Chlorophyll|leechseed,substitute,sleeppowder,sludgebomb||85,,85,85,85,85||,0,,,,||82|]Blastoise||WhiteHerb|Torrent|shellsmash,earthquake,icebeam,hydropump||85,85,85,85,85,85||||86|"
# TEAM = "Ceruledge||LifeOrb|WeakArmor|bitterblade,closecombat,shadowsneak,swordsdance||85,85,85,85,85,85||||82|,,,,,Fighting]Grafaiai||Leftovers|Prankster|encore,gunkshot,knockoff,partingshot||85,85,85,85,85,85||||86|,,,,,Dark]Greedent||SitrusBerry|CheekPouch|bodyslam,psychicfangs,swordsdance,firefang||85,85,85,85,85,85||||88|,,,,,Psychic]Quaquaval||LifeOrb|Moxie|aquastep,closecombat,swordsdance,icespinner||85,85,85,85,85,85||||80|,,,,,Fighting]Flapple||LifeOrb|Hustle|gravapple,outrage,dragondance,suckerpunch||85,85,85,85,85,85||||84|,,,,,Grass]Pachirisu||AssaultVest|VoltAbsorb|nuzzle,superfang,thunderbolt,uturn||85,85,85,85,85,85||||94|,,,,,Flying"


def main():
    # controller = RandomController()
    controller = NaiveAIController()

    procs = []
    for i in range(1):  # num workes (check with os.cpu_count())
        worker = SelfPlayWorker(i, 2, BATTLE_FORMAT, TEAM)  # 2 is players per worker
        # This config will spawn 20 workers with 2 players each
        # for a total of 40 players, playing 20 games.
        # it is recommended to have an even number of players per worker

        process = mp.Process(
            target=worker.run,
            args=(controller,),
            name=repr(worker),
        )
        process.start()
        procs.append(process)

    for proc in procs:
        proc.join()


if __name__ == "__main__":
    mp.set_start_method("fork")
    main()
