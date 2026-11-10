"""
Versión 2: MEMORIA COMPARTIDA / procesos locales con multiprocessing.Pool y joblib.

Estrategias de reparto:
  - estatico          : p bloques contiguos (pool.map con chunksize = ceil(N/p)) -> reparto fijo de antemano
  - dinamico          : cada proceso pide la siguiente tarea al terminar (imap_unordered, chunksize=1)
  - dinamico_ordenado : como el dinámico, pero entrega primero las tareas estimadas como más pesadas (tipo LPT)
"""
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "comun"))
from worker import evaluate, init_worker, reset_data
from utils_hpo import asignar_worker, costo_estimado


def run_pool(strategy, p, tasks, data_path):
    tasks = list(tasks)
    if strategy == "dinamico_ordenado":
        tasks.sort(key=lambda t: costo_estimado(t[1]), reverse=True)
    reset_data()                                # el proceso padre no debe "regalar" datos a los hijos (fork)
    t0_epoch, t0 = time.time(), time.perf_counter()
    with mp.Pool(processes=p, initializer=init_worker, initargs=(str(data_path),)) as pool:
        if strategy == "estatico":
            cs = math.ceil(len(tasks) / p)      # p bloques contiguos
            res = pool.map(evaluate, tasks, chunksize=cs)
        else:
            res = list(pool.imap_unordered(evaluate, tasks, chunksize=1))
    wall = time.perf_counter() - t0
    return wall, t0_epoch, asignar_worker(pd.DataFrame(res))


def run_joblib(strategy, p, tasks):
    """Mismo trabajo con joblib: batch_size grande ~ estático; batch_size=1 ~ dinámico."""
    from joblib import Parallel, delayed
    from joblib.externals.loky import get_reusable_executor
    get_reusable_executor().shutdown(wait=True)   # los trabajadores no deben conservar datos de la corrida anterior
    reset_data()
    bs = math.ceil(len(tasks) / p) if strategy == "estatico" else 1
    t0_epoch, t0 = time.time(), time.perf_counter()
    res = Parallel(n_jobs=p, batch_size=bs, backend="loky")(delayed(evaluate)(t) for t in tasks)
    wall = time.perf_counter() - t0
    return wall, t0_epoch, asignar_worker(pd.DataFrame(res))
