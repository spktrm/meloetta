import random
import asyncio

import multiprocessing as mp

from meloetta.player import Player
from typing import Any
from tqdm import tqdm


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

        battle_format = "gen1randombattle"
        team = "null"

        progress = tqdm(range(10)) if player_index == 0 else range(10)
        for _ in progress:  # 10 battles each player-player pair
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
                    if "Can't switch: The active Pokémon is trapped" in message:
                        message = await player.client.receive_message()
                        action_required = await player.recieve(message)
                        action_required = True
                    else:
                        print(message)

                if action_required:
                    state = player.get_state()
                    choices = player.get_choices()
                    # The magic
                    choice = (
                        "/choose " + random.choice(choices)
                        if choices
                        else "/choose default"
                    )
                    msg_list = [choice, player.rqid]
                    await player.client.send_message(player.battle.battle_tag, msg_list)

                if player.battle.ended:
                    await player.client.leave_battle(player.battle.battle_tag)
                    player.reset()
                    break


def main():
    procs = []
    for i in range(20): # num workes (check with os.cpu_count())
        worker = SelfPlayWorker(i, 1) # 2 is players per worker
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
