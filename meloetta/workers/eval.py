import torch
import asyncio

from typing import Any, Sequence, Mapping, Type

from meloetta.player import Player
from meloetta.room import BattleRoom
from meloetta.actors.base import Actor
from meloetta.workers.barrier import Barrier

from meloetta.room import BattleRoom
from meloetta.utils import expand_bt


def waiting_for_opp(room: BattleRoom):
    controls = room.get_js_attr("controls.controls")
    return (
        "Waiting for opponent" in controls or " will switch in, replacing" in controls
    )


DRAW_BY_TURNS = 200


class EvalWorker:
    def __init__(
        self,
        eval_username: str,
        opponent_username: str,
        battle_format: str,
        team: str,
        eval_actor_fn: Type[Actor] = None,
        eval_actor_args: Sequence[Any] = None,
        eval_actor_kwargs: Mapping[str, Any] = None,
        baseline_actor_fn: Type[Actor] = None,
        baseline_actor_args: Sequence[Any] = None,
        baseline_actor_kwargs: Mapping[str, Any] = None,
        sleep: float = None,
    ):
        self.battle_format = battle_format
        self.team = team

        self.eval_username = eval_username
        self.opponent_username = opponent_username

        self.eval_actor_fn = eval_actor_fn
        self.baseline_actor_fn = baseline_actor_fn

        self.eval_actor_args = () if eval_actor_args is None else eval_actor_args
        self.eval_actor_kwargs = {} if eval_actor_kwargs is None else eval_actor_kwargs

        self.baseline_actor_args = (
            () if baseline_actor_args is None else baseline_actor_args
        )
        self.baseline_actor_kwargs = (
            {} if baseline_actor_kwargs is None else baseline_actor_kwargs
        )
        self.sleep = sleep

    def __repr__(self) -> str:
        return f"EvalWorker"

    def run(self) -> Any:
        """
        Start selfplay between two asynchronous actors-
        """

        async def selfplay():
            barrier = Barrier(2)
            return await asyncio.gather(
                self.actor(0, self.eval_actor_fn, barrier),
                self.actor(1, self.baseline_actor_fn, barrier),
            )

        results = asyncio.run(selfplay())
        return results

    async def actor(
        self, player_index: int, actor_fn: Type[Actor], barrier: Barrier
    ) -> Any:
        if player_index == 0:
            username = self.eval_username
        else:
            username = self.opponent_username

        player = await Player.create(username, None, "localhost:8000")
        await player.client.login()
        await barrier.wait()

        while True:  # 10 battles each player-player pair
            if self.sleep:
                await asyncio.sleep(2)

            if player_index == 0:
                await player.client.challenge_user(
                    self.opponent_username, self.battle_format, self.team
                )
                actor = actor_fn(*self.eval_actor_args, **self.eval_actor_kwargs)
            else:
                await player.client.accept_challenge(self.battle_format, self.team)
                actor = actor_fn(
                    *self.baseline_actor_args, **self.baseline_actor_kwargs
                )

            turn = 0
            turns_since_last_move = expand_bt(torch.tensor(0))

            while True:
                message = await player.client.receive_message()
                action_required = await player.recieve(message)
                if "is offering a tie." in message:
                    await player.client.websocket.send(
                        player.room.battle_tag + "|" + "/offertie"
                    )
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
