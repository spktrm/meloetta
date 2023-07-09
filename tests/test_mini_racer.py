import multiprocessing as mp
import threading as t

from py_mini_racer import MiniRacer

N = 20


def p(id):
    ctx1 = MiniRacer()
    ctx1.eval("var x = 0")
    x1 = 0
    ctx2 = MiniRacer()
    ctx2.eval("var x = 0")
    x2 = 0
    for _ in range(10000):
        ctx1_v = ctx1.eval("x++; x")
        x1 += 1
        assert x1 == int(ctx1_v)
        ctx2_v = ctx2.eval("x--; x")
        x2 -= 1
        assert x2 == int(ctx2_v)


def main_proc():
    procs = []
    for i in range(N):
        proc = mp.Process(target=p, args=(i,))
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join()

    return True


def main_thread():
    procs = []
    for i in range(N):
        proc = t.Thread(target=p, args=(i,))
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join()

    return True


def test_mini_racer():
    assert main_proc()
    assert main_thread()
