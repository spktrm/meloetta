import os

os.environ["OMP_NUM_THREADS"] = "1"

import wandb
import threading
import multiprocessing as mp

from typing import List

from tqdm import tqdm

from meloetta.workers import SelfPlayWorker, EvalWorker

from meloetta.frameworks.random import RandomActor
from meloetta.frameworks.max_damage import MaxDamageActor
from meloetta.frameworks.proxima import ProximaActor, ProximaLearner


def start_frame_monitor(queue: mp.Queue):
    frame_monitor = tqdm(desc="frames: ")
    episode_monitor = tqdm(desc="eps: ")
    while True:
        _, frames = queue.get()
        frame_monitor.update(frames)
        episode_monitor.update(1)
        if not frames:
            break


def start_eval_monitor(queue: mp.Queue):
    random = 0
    maxdmg = 0
    while True:
        baseline, res = queue.get()
        if baseline == "random":
            n = random
            random += 1
        elif baseline == "maxdmg":
            n = maxdmg
            maxdmg += 1
        wandb.log({f"{baseline}n": n, baseline: 1 - res})


def main(fpath: str = None, latest: bool = False):
    if latest:
        print("Using latest save checkpoint")
        learner = ProximaLearner.from_latest()

    elif fpath:
        print(f"Loading run from: {fpath}")
        learner = ProximaLearner.from_fpath(fpath)

    else:
        learner = ProximaLearner()

    wandb.init(
        project="meloetta",
        config=learner.get_config(),
    )

    eval_queue = mp.Queue()

    main_actor = ProximaActor
    random_actor = RandomActor
    maxdmg_actor = MaxDamageActor

    procs: List[mp.Process] = []

    threads: List[threading.Thread] = []

    frame_monitor_thread = threading.Thread(
        target=start_frame_monitor, args=(learner.replay_buffer.finish_queue,)
    )
    frame_monitor_thread.start()
    threads.append(frame_monitor_thread)

    for i in range(1):
        learner_thread = threading.Thread(
            target=learner.run,
            name=f"Learning Thread{i}",
        )
        threads.append(learner_thread)
        learner_thread.start()

    # if learner.config.debug_mode:
    num_players = 2
    # else:
    #     num_players = max(learner.config.batch_size // learner.config.num_actors, 2)

    for i in range(learner.config.num_actors):
        worker = SelfPlayWorker(
            worker_index=i,
            num_players=num_players,  # 2 is players per worker
            battle_format=learner.config.battle_format,
            team=learner.config.team,
            actor_fn=main_actor,
            actor_kwargs={
                "model": learner.actor_model,
                "replay_buffer": learner.replay_buffer,
            },
        )

        process = mp.Process(
            target=worker.run,
            name=repr(worker) + str(i),
        )
        process.start()
        procs.append(process)

    if learner.config.eval:
        evals = [
            (f"random_eval{i}", f"random{i}", random_actor, (eval_queue,))
            for i in range(0)
        ] + [
            (
                f"maxdmg_eval{i}",
                f"maxdmg{i}",
                maxdmg_actor,
                (learner.config.gen, eval_queue),
            )
            for i in range(1)
        ]
        for i, (
            eval_username,
            opponent_username,
            opponent_actor,
            opponent_actor_args,
        ) in enumerate(evals):
            worker = EvalWorker(
                eval_username=eval_username,
                opponent_username=opponent_username,
                battle_format=learner.config.battle_format,
                team=learner.config.team,
                eval_actor_fn=main_actor,
                eval_actor_kwargs={
                    "model": learner.actor_model,
                    # "replay_buffer": learner.replay_buffer,
                },
                baseline_actor_fn=opponent_actor,
                baseline_actor_args=opponent_actor_args,
            )
            process = mp.Process(
                target=worker.run,
                name=repr(worker) + str(i),
            )
            process.start()
            procs.append(process)

    eval_monitor_thread = threading.Thread(
        target=start_eval_monitor, args=(eval_queue,)
    )
    eval_monitor_thread.start()
    threads.append(eval_monitor_thread)

    for thread in threads:
        thread.join()

    for proc in procs:
        proc.join()


if __name__ == "__main__":
    mp.set_start_method("spawn")

    # fpath = "cpkts/cpkt-00001000.tar"
    # main(fpath)

    # main(latest=True)

    main()
