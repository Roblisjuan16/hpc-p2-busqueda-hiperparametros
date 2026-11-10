"""Versión 1: SECUENCIAL (línea base). Un ciclo `for` evalúa las combinaciones una por una."""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "comun"))
from worker import evaluate, init_worker, reset_data


def run_secuencial(tasks, data_path):
    """Devuelve (tiempo_total_s, instante_inicio_epoch, DataFrame con un renglón por tarea)."""
    reset_data()
    t0_epoch, t0 = time.time(), time.perf_counter()
    init_worker(data_path)                      # el tiempo incluye la carga del dataset (igual que en las paralelas)
    resultados = []
    for tarea in tasks:                         # <-- el ciclo for secuencial
        resultados.append(evaluate(tarea))
    wall = time.perf_counter() - t0
    df = pd.DataFrame(resultados)
    df["worker"] = 0
    return wall, t0_epoch, df
