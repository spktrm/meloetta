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
from meloetta.frameworks.nash_ketchum import (
    NAshKetchumActor,
    NAshKetchumLearner,
    NAshKetchumConfig,
)


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


def main(fpath: str = None):
    config = NAshKetchumConfig()

    if fpath is None:
        print(f"Starting run with config:\n{repr(config)}")
        learner = NAshKetchumLearner.from_config(config)
    else:
        print(f"Loading run from: {fpath}")
        learner = NAshKetchumLearner.from_pretrained(fpath, config)

    wandb.init(
        project="meloetta",
        config=learner.get_config(),
    )

    main_actor = NAshKetchumActor(learner.actor_model, learner.replay_buffer)

    eval_queue = mp.Queue()
    random_actor = RandomActor(eval_queue)
    maxdmg_actor = MaxDamageActor(main_actor._model.gen, eval_queue)

    procs: List[mp.Process] = []

    if config.eval:
        evals = [
            ("eval0", "random", random_actor),
            ("eval1", "maxdmg", maxdmg_actor),
        ]
        for i, (eval_username, opponent_username, opponent_actor) in enumerate(evals):
            worker = EvalWorker(
                eval_username=eval_username,
                opponent_username=opponent_username,
                battle_format=learner.config.battle_format,
                team=learner.config.team,
            )
            process = mp.Process(
                target=worker.run,
                args=(main_actor, opponent_actor),
                name=repr(worker) + str(i),
            )
            process.start()
            procs.append(process)

    threads: List[threading.Thread] = []
    for i in range(1):
        learner_thread = threading.Thread(
            target=learner.run,
            name=f"Learning Thread{i}",
        )
        threads.append(learner_thread)
        learner_thread.start()

    for i in range(config.num_actors):
        worker = SelfPlayWorker(
            worker_index=i,
            num_players=2,  # 2 is players per worker
            battle_format=learner.config.battle_format,
            team=learner.config.team,
        )

        process = mp.Process(
            target=worker.run,
            args=(main_actor,),
            name=repr(worker) + str(i),
        )
        process.start()
        procs.append(process)

    frame_monitor_thread = threading.Thread(
        target=start_frame_monitor, args=(learner.replay_buffer.finish_queue,)
    )
    frame_monitor_thread.start()
    threads.append(frame_monitor_thread)

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

    # fpath = "cpkts/cpkt-09000.tar"
    # main(fpath)

    main()
