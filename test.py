import random
import asyncio

import multiprocessing as mp

from typing import Any
from tqdm import tqdm

from meloetta.player import Player
from meloetta.room import BattleRoom
from meloetta.vector import VectorizedState, VectorizedChoice


def waiting_for_opp(room: BattleRoom):
    controls = room.get_js_attr("controls.controls")
    return (
        "Waiting for opponent" in controls or " will switch in, replacing" in controls
    )


class SelfPlayWorker:
    def __init__(
        self,
        worker_index: int,
        num_players: int,
    ):
        self.worker_index = worker_index
        self.num_players = num_players

    def __repr__(self) -> str:
        return f"Worker{self.worker_index}"

    def run(self) -> Any:
        """
        Start selfplay between two asynchronous actors-
        """

        async def selfplay():
            return await asyncio.gather(
                *[
                    self.actor(self.worker_index * self.num_players + i)
                    for i in range(self.num_players)
                ]
            )

        results = asyncio.run(selfplay())
        return results

    async def actor(self, player_index: int) -> Any:
        username = f"p{player_index}"

        player = await Player.create(username, None, "localhost:8000")
        await player.client.login()
        await asyncio.sleep(1)

        battle_format = "gen8randomdoublesbattle"
        team = "null"
        # battle_format = "gen9doublesou"
        # team = "Charizard||HeavyDutyBoots|Blaze|hurricane,fireblast,toxic,roost||85,,85,85,85,85||,0,,,,||88|]Venusaur||BlackSludge|Chlorophyll|leechseed,substitute,sleeppowder,sludgebomb||85,,85,85,85,85||,0,,,,||82|]Blastoise||WhiteHerb|Torrent|shellsmash,earthquake,icebeam,hydropump||85,85,85,85,85,85||||86|"
        # team = "Ceruledge||LifeOrb|WeakArmor|bitterblade,closecombat,shadowsneak,swordsdance||85,85,85,85,85,85||||82|,,,,,Fighting]Grafaiai||Leftovers|Prankster|encore,gunkshot,knockoff,partingshot||85,85,85,85,85,85||||86|,,,,,Dark]Greedent||SitrusBerry|CheekPouch|bodyslam,psychicfangs,swordsdance,firefang||85,85,85,85,85,85||||88|,,,,,Psychic]Quaquaval||LifeOrb|Moxie|aquastep,closecombat,swordsdance,icespinner||85,85,85,85,85,85||||80|,,,,,Fighting]Flapple||LifeOrb|Hustle|gravapple,outrage,dragondance,suckerpunch||85,85,85,85,85,85||||84|,,,,,Grass]Pachirisu||AssaultVest|VoltAbsorb|nuzzle,superfang,thunderbolt,uturn||85,85,85,85,85,85||||94|,,,,,Flying"

        for _ in range(100):  # 10 battles each player-player pair
            if player_index % 2 == 0:
                await player.client.challenge_user(
                    f"p{player_index + 1}", battle_format, team
                )
            else:
                await player.client.accept_challenge(battle_format, team)

            while True:
                message = await player.client.receive_message()
                action_required = await player.recieve(message)
                if "|error" in message:
                    # print(message)
                    # edge case for handling when the pokemon is trapped
                    if "Can't switch: The active Pokémon is trapped" in message:
                        message = await player.client.receive_message()
                        action_required = await player.recieve(message)
                        action_required = True

                    # for some reason, disabled max moves are being selected
                    elif "Can't move" in message:
                        print()

                if action_required:
                    # inputs to neural net
                    battle = player.room.get_battle()
                    vstate = VectorizedState.from_battle(player.room, battle)
                    vchoice = VectorizedChoice.from_battle(player.room, battle)

                while (
                    action_required
                    and not waiting_for_opp(player.room)
                    and not player.room.get_js_attr("battle.ended")
                ):
                    choices = player.get_choices()
                    _, func, args, kwargs = random.choice(choices)
                    func(*args, **kwargs)

                outgoing_message = player.room.get_js_attr("outgoing_message")
                if outgoing_message:
                    player.room.pop_outgoing()
                    await player.client.websocket.send(
                        player.room.battle_tag + "|" + outgoing_message
                    )

                if player.room.get_js_attr("battle.ended"):
                    await player.client.leave_battle(player.room.battle_tag)
                    player.reset()
                    break

        print(f"{username} done")


def main():
    procs = []
    for i in range(1):  # num workes (check with os.cpu_count())
        worker = SelfPlayWorker(i, 2)  # 2 is players per worker
        # This config will spawn 20 workers with 2 players each
        # for a total of 40 players, playing 20 games.
        # it is recommended to have an even number of players per worker

        process = mp.Process(
            target=worker.run,
            name=repr(worker),
        )
        process.start()
        procs.append(process)

    for proc in procs:
        proc.join()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
