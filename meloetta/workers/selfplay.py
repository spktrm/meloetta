import torch
import asyncio

from typing import Any, Sequence, Mapping, Type

from meloetta.player import Player
from meloetta.room import BattleRoom
from meloetta.actors.base import Actor
from meloetta.workers.barrier import Barrier

from meloetta.room import BattleRoom
from meloetta.utils import expand_bt

DRAW_BY_TURNS = 200


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
        actor_fn: Type[Actor],
        actor_args: Sequence[Any] = None,
        actor_kwargs: Mapping[str, Any] = None,
    ):
        self.battle_format = battle_format
        self.team = team

        self.worker_index = worker_index
        self.num_players = num_players

        self.actor_fn = actor_fn
        self.actor_args = () if actor_args is None else actor_args
        self.actor_kwargs = {} if actor_kwargs is None else actor_kwargs

    def __repr__(self) -> str:
        return f"Worker{self.worker_index}"

    def run(self) -> Any:
        """
        Start selfplay between two asynchronous actors-
        """

        async def selfplay():
            barrier = Barrier(self.num_players)
            return await asyncio.gather(
                *[
                    self.actor(self.worker_index * self.num_players + i, barrier)
                    for i in range(self.num_players)
                ]
            )

        results = asyncio.run(selfplay())
        return results

    async def actor(self, player_index: int, barrier: Barrier) -> Any:
        username = f"player{player_index}"

        player = await Player.create(username, None, "localhost:8000")
        await player.client.login()
        await barrier.wait()

        while True:  # 10 battles each player-player pair
            if player_index % 2 == 0:
                await player.client.challenge_user(
                    f"player{player_index + 1}", self.battle_format, self.team
                )
            else:
                await player.client.accept_challenge(self.battle_format, self.team)

            turn = 0
            turns_since_last_move = expand_bt(torch.tensor(0))
            actor = self.actor_fn(
                *self.actor_args, **self.actor_kwargs, username=username
            )

            while True:
                message = await player.client.receive_message()
                action_required = await player.recieve(message)
                if "is offering a tie." in message:
                    await player.client.websocket.send(
                        player.room.battle_tag + "|" + "/offertie"
                    )
                if "|error" in message:
                    # edge case for handling when the pokemon is trapped
                    if "Can't switch: The active Pokémon is trapped" in message:
                        message = await player.client.receive_message()
                        action_required = await player.recieve(message)
                        action_required = True

                    # for some reason, disabled max moves are being selected
                    elif "Can't move" in message:
                        print(message)

                    else:
                        print(message)

                if action_required:
                    # inputs to neural net
                    battle = player.room.get_battle()
                    turn = battle["turn"]
                    vstate = actor.get_vectorized_state(player.room, battle)

                ended = player.room.get_js_attr("battle?.ended")
                while (
                    action_required and not waiting_for_opp(player.room) and not ended
                ):
                    choices = player.get_choices()
                    state = {
                        **vstate,
                        "turns_since_last_move": turns_since_last_move,
                        **choices.action_masks,
                        **choices.prev_choices,
                        "targeting": choices.targeting,
                    }

                    func, args, kwargs = actor(state, player.room, choices.choices)
                    func(*args, **kwargs)

                outgoing_message = player.room.get_js_attr("outgoing_message")
                if outgoing_message:
                    player.room.pop_outgoing()
                    if "move" in outgoing_message:
                        turns_since_last_move *= 0
                    else:
                        turns_since_last_move += 1
                    await player.client.websocket.send(
                        player.room.battle_tag + "|" + outgoing_message
                    )

                if ended:
                    break

                if turns_since_last_move > 50:
                    print(f"{username}: forfeit by repetition!")

                    await player.client.websocket.send(
                        player.room.battle_tag + "|" + "/forfeit"
                    )
                    while True:
                        message = await player.client.receive_message()
                        action_required = await player.recieve(message)
                        ended = player.room.get_js_attr("battle?.ended")
                        if ended:
                            break
                    break

                if turn > 200:
                    print(f"{username}: draw by turn > {DRAW_BY_TURNS}!")

                    await player.client.websocket.send(
                        player.room.battle_tag + "|" + "/offertie"
                    )
                    while True:
                        message = await player.client.receive_message()
                        action_required = await player.recieve(message)
                        if "is offering a tie." in message:
                            await player.client.websocket.send(
                                player.room.battle_tag + "|" + "/offertie"
                            )
                        ended = player.room.get_js_attr("battle?.ended")
                        if ended:
                            break
                    break

            await player.client.leave_battle(player.room.battle_tag)
            actor.post_match(player.room)
            player.reset()
