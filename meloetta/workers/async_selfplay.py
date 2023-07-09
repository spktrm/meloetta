import torch
import asyncio

from typing import Any, Sequence, Mapping, Type

from meloetta.player import Player
from meloetta.room import BattleRoom
from meloetta.actors.base import Actor
from meloetta.workers.barrier import Barrier

from meloetta.types import TensorDict
from meloetta.room import BattleRoom
from meloetta.utils import expand_bt
from meloetta.frameworks.nash_ketchum.buffer import create_buffers

DRAW_BY_TURNS = 300


def waiting_for_opp(room: BattleRoom):
    controls = room.get_js_attr("controls.controls")
    return (
        "Waiting for opponent" in controls or " will switch in, replacing" in controls
    )


class InferenceBuffer:
    def __init__(self, model: torch.nn.Module, batch_size: int, device: str):
        self.model = model
        self.device = device

        self.full_queue = asyncio.Queue()
        self.free_queue = asyncio.Queue()

        self.batch_size = batch_size
        self.buffer = create_buffers(2 * batch_size, 1, 9, "singles", 1)

        # Start the background task
        self.bg_task = asyncio.create_task(self.process_queue())

    async def process_queue(self):
        for m in range(2 * self.batch_size):
            await self.free_queue.put(m)
        indices = []
        futures = []
        while True:
            while len(indices) < self.batch_size:
                try:
                    future, index = await asyncio.wait_for(self.full_queue.get(), 0.1)
                except asyncio.TimeoutError:
                    # The wait_for call timed out, so we process the current batch
                    break
                else:
                    # We don't have enough items for a batch yet, so we wait
                    indices.append(index)
                    futures.append(future)

            if indices:
                with torch.no_grad():
                    await self.process_batch(indices, futures)
                indices = []
                futures = []

    async def process_batch(self, indices, futures):
        # This method should be implemented to process a batch of items and produce results
        loop = asyncio.get_event_loop()
        batch = {
            key: torch.stack([self.buffer[key][0][index] for index in indices]).to(
                self.device
            )
            for key in self.buffer
            if not (key.endswith("_index") or "policy" in key)
        }
        results = await loop.run_in_executor(None, self.model, batch)
        # results = self.model(batch)

        results = [
            {key: results[key][n].cpu() for key in results.keys()}
            for n in range(len(futures))
        ]

        assert len(results) == len(futures)
        for index, future, result in zip(indices, futures, results):
            future.set_result(result)
            await self.free_queue.put(index)

    async def submit(self, sample: TensorDict) -> asyncio.Future:
        future = asyncio.get_event_loop().create_future()
        index = await self.free_queue.get()
        for key, value in sample.items():
            if value is not None and key in self.buffer:
                self.buffer[key][0][index][...] = value
        await self.full_queue.put((future, index))
        return future


class AsyncSelfPlayWorker:
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

        self.inference_buffer = InferenceBuffer(
            model=self.actor_kwargs["model"], batch_size=64, device="cpu"
        )

    def __repr__(self) -> str:
        return f"Worker{self.worker_index}"

    async def selfplay(self):
        barrier = Barrier(self.num_players)
        return await asyncio.gather(
            *[
                self.actor(self.worker_index * self.num_players + i, barrier)
                for i in range(self.num_players)
            ]
        )

    async def start_battle(self, player: Player, player_index: int):
        if player_index % 2 == 0:
            await player.client.challenge_user(
                f"player{player_index + 1}", self.battle_format, self.team
            )
        else:
            await player.client.accept_challenge(self.battle_format, self.team)

    async def actor(self, player_index: int, barrier: Barrier) -> Any:
        username = f"player{player_index}"

        player = await Player.create(username, None, "localhost:8000")
        await player.client.login()
        await barrier.wait()

        while True:
            await self.start_battle(player, player_index)

            turn = 0
            turns_since_last_move = expand_bt(torch.tensor(0, dtype=torch.long))
            actor = self.actor_fn(
                *self.actor_args,
                **self.actor_kwargs,
                pid=(0 if player_index % 2 == 0 else 1),
            )

            while True:
                message = await player.client.receive_message()
                action_required = player._recieve(message)

                if "is offering a tie." in message:
                    await player.client.websocket.send(
                        player.room.battle_tag + "|" + "/offertie"
                    )

                if "|error" in message:
                    # edge case for handling when the pokemon is trapped
                    if "Can't switch: The active Pokémon is trapped" in message:
                        message = await player.client.receive_message()
                        action_required = player._recieve(message)
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
                    choices = player.get_choices(turns_since_last_move)
                    state = {
                        **vstate,
                        **choices.action_masks,
                        **choices.prev_choices,
                        "targeting": choices.targeting,
                    }

                    future = await self.inference_buffer.submit(state)
                    model_output = await future
                    func, args, kwargs = actor.post_process(
                        state, model_output, player.room, choices.choices
                    )
                    func(*args, **kwargs)

                outgoing_message = player.room.get_js_attr("outgoing_message")
                if outgoing_message:
                    player.room.pop_outgoing()
                    if "move" in outgoing_message:
                        turns_since_last_move = turns_since_last_move * 0
                    else:
                        turns_since_last_move = turns_since_last_move + 1
                    await player.client.websocket.send(
                        player.room.battle_tag + "|" + outgoing_message
                    )

                if ended:
                    break

                if turn > DRAW_BY_TURNS:
                    print(f"{username}: draw by turn > {DRAW_BY_TURNS}!")

                    await player.client.websocket.send(
                        player.room.battle_tag + "|" + "/offertie"
                    )
                    while True:
                        message = await player.client.receive_message()
                        action_required = player._recieve(message)

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
