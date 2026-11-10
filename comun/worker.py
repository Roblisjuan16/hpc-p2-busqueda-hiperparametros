"""
Función de trabajo COMPARTIDA por todas las versiones (secuencial, Pool, joblib, MPI, Ray).

Cada tarea = una combinación de hiperparámetros evaluada con validación cruzada estratificada.
Usar exactamente la misma función en todas las versiones garantiza una comparación justa y
permite verificar la correctitud (mismos puntajes en todas las versiones).
"""
import os
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

_X = None
_y = None


def reset_data():
    """Olvida los datos cargados, para que cada corrida cargue el dataset desde cero."""
    global _X, _y
    _X, _y = None, None


def _get_data(path):
    """Carga el dataset una sola vez por proceso y lo deja en una variable global."""
    global _X, _y
    if _X is None:
        d = np.load(path)
        _X, _y = d["X"], d["y"]
    return _X, _y


def init_worker(path):
    """Inicializador de proceso: carga los datos al arrancar (fuera del tiempo de cada tarea)."""
    _get_data(path)


def evaluate(task):
    """Evalúa UNA combinación de hiperparámetros. Devuelve puntajes, tiempos y el PID del proceso."""
    idx, params, path, folds, seed = task
    X, y = _get_data(path)
    t0 = time.time()
    clf = RandomForestClassifier(random_state=seed, n_jobs=1, **params)   # n_jobs=1: el paralelismo es ENTRE tareas
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    s = cross_validate(clf, X, y, cv=cv, scoring=("accuracy", "f1_macro"), n_jobs=1)
    t1 = time.time()
    out = dict(idx=idx, **params)
    out.update(acc=float(s["test_accuracy"].mean()), f1=float(s["test_f1_macro"].mean()),
               t_start=t0, t_end=t1, dur=t1 - t0, pid=os.getpid())
    return out
