import asyncio
import threading

from typing import Any, List

from meloetta.player import Player
from meloetta.room import BattleRoom
from meloetta.vector import VectorizedState
from meloetta.actors.base import Actor
from meloetta.workers.barrier import Barrier


def waiting_for_opp(room: BattleRoom):
    controls = room.get_js_attr("controls.controls")
    return (
        "Waiting for opponent" in controls or " will switch in, replacing" in controls
    )


class EvalWorker:
    def __init__(
        self,
        eval_username: str,
        opponent_username: str,
        battle_format: str,
        team: str,
    ):
        self.battle_format = battle_format
        self.team = team

        self.eval_username = eval_username
        self.opponent_username = opponent_username

    def __repr__(self) -> str:
        return f"EvalWorker"

    def run(self, eval_actor: Actor, baseline_actor: Actor) -> Any:
        """
        Start selfplay between two asynchronous actors-
        """

        async def selfplay(eval_actor, baseline_actor):
            barrier = Barrier(2)
            return await asyncio.gather(
                self.actor(0, eval_actor, barrier),
                self.actor(1, baseline_actor, barrier),
            )

        results = asyncio.run(selfplay(eval_actor, baseline_actor))
        return results

    async def actor(self, player_index: int, actor: Actor, barrier: Barrier) -> Any:
        if player_index == 0:
            username = self.eval_username
        else:
            username = self.opponent_username

        player = await Player.create(username, None, "localhost:8000")
        await player.client.login()
        await barrier.wait()

        while True:  # 10 battles each player-player pair
            await asyncio.sleep(2)
            if player_index == 0:
                await player.client.challenge_user(
                    self.opponent_username, self.battle_format, self.team
                )
            else:
                await player.client.accept_challenge(self.battle_format, self.team)

            turn = 0
            turns_since_last_move = 0
            while True:
                message = await player.client.receive_message()
                if "is offering a tie." in message:
                    await player.client.websocket.send(
                        player.room.battle_tag + "|" + "/offertie"
                    )
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
                    turn = battle["turn"]
                    vstate = VectorizedState.from_battle(player.room, battle)

                ended = player.room.get_js_attr("battle?.ended")
                while (
                    action_required and not waiting_for_opp(player.room) and not ended
                ):
                    choices = player.get_choices()
                    state = {
                        **vstate.to_dict(turns_since_last_move),
                        **choices.action_masks,
                        **choices.prev_choices,
                        "targeting": choices.targeting,
                    }

                    func, args, kwargs = actor(
                        state, player.room, choices.choices, store_transition=False
                    )
                    func(*args, **kwargs)

                outgoing_message = player.room.get_js_attr("outgoing_message")
                if outgoing_message:
                    player.room.pop_outgoing()
                    if "move" in outgoing_message:
                        turns_since_last_move = 0
                    else:
                        turns_since_last_move += 1
                    await player.client.websocket.send(
                        player.room.battle_tag + "|" + outgoing_message
                    )

                if ended:
                    break

                if (turns_since_last_move > 50 and turn > 100) or turn > 200:
                    if turns_since_last_move > 50:
                        print(f"{username}: draw by repetition!")
                    elif turn > 100:
                        print(f"{username}: draw by turn > 200!")
                    await player.client.websocket.send(
                        player.room.battle_tag + "|" + "/offertie"
                    )
                    while True:
                        message = await player.client.receive_message()
                        if "is offering a tie." in message:
                            await player.client.websocket.send(
                                player.room.battle_tag + "|" + "/offertie"
                            )
                        action_required = await player.recieve(message)
                        ended = player.room.get_js_attr("battle?.ended")
                        if ended:
                            break
                    break

            await player.client.leave_battle(player.room.battle_tag)
            datum = player.room.get_reward()
            actor.store_reward(
                player.room, datum["pid"], datum["reward"], store_transition=False
            )
            player.reset()
