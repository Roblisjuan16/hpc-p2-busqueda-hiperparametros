"""Alternativa opcional: Ray. Todas las tareas se lanzan como futures y los trabajadores las toman bajo demanda."""
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "comun"))
from worker import evaluate, reset_data
from utils_hpo import asignar_worker


def run_ray(p, tasks):
    import ray
    reset_data()
    t0_epoch, t0 = time.time(), time.perf_counter()
    ray.init(num_cpus=p, include_dashboard=False, logging_level="ERROR",
             runtime_env={"env_vars": {"PYTHONPATH": os.environ.get("PYTHONPATH", "")}})
    remote_eval = ray.remote(evaluate)
    res = ray.get([remote_eval.remote(t) for t in tasks])
    ray.shutdown()
    wall = time.perf_counter() - t0
    return wall, t0_epoch, asignar_worker(pd.DataFrame(res))
