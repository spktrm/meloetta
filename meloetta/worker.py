import asyncio

from typing import Any
from tqdm import tqdm

from meloetta.player import Player
from meloetta.room import BattleRoom
from meloetta.vector import VectorizedState
from meloetta.controllers.base import Controller


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
        battle_format: str,
        team: str,
    ):
        self.battle_format = battle_format
        self.team = team
        self.worker_index = worker_index
        self.num_players = num_players

    def __repr__(self) -> str:
        return f"Worker{self.worker_index}"

    def run(self, controller: Controller) -> Any:
        """
        Start selfplay between two asynchronous actors-
        """

        async def selfplay(controller: Controller):
            return await asyncio.gather(
                *[
                    self.actor(self.worker_index * self.num_players + i, controller)
                    for i in range(self.num_players)
                ]
            )

        results = asyncio.run(selfplay(controller))
        return results

    async def actor(self, player_index: int, controller: Controller) -> Any:
        username = f"p{player_index}"

        player = await Player.create(username, None, "localhost:8000")
        await player.client.login()
        await asyncio.sleep(1)

        for _ in range(1000):  # 10 battles each player-player pair
            if player_index % 2 == 0:
                await player.client.challenge_user(
                    f"p{player_index + 1}", self.battle_format, self.team
                )
            else:
                await player.client.accept_challenge(self.battle_format, self.team)

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
                        print(message)

                if action_required:
                    # inputs to neural net
                    battle = player.room.get_battle()
                    vstate = VectorizedState.from_battle(player.room, battle)

                ended = player.room.get_js_attr("battle?.ended")
                while (
                    action_required and not waiting_for_opp(player.room) and not ended
                ):
                    choices = player.get_choices()
                    state = {
                        **vstate.to_dict(),
                        **choices.action_masks,
                        **choices.prev_choices,
                        "targeting": choices.targeting,
                    }

                    func, args, kwargs = controller(state, player.room, choices.choices)
                    func(*args, **kwargs)

                outgoing_message = player.room.get_js_attr("outgoing_message")
                if outgoing_message:
                    player.room.pop_outgoing()
                    await player.client.websocket.send(
                        player.room.battle_tag + "|" + outgoing_message
                    )

                if ended:
                    await player.client.leave_battle(player.room.battle_tag)
                    datum = player.room.get_reward()
                    controller.store_reward(player.room, datum["pid"], datum["reward"])
                    player.reset()
                    break
