"""Configuración del experimento, generación de tareas y registro de corridas."""
import json
import math
from pathlib import Path

import pandas as pd
from sklearn.model_selection import ParameterGrid

# "prueba"   -> pocos datos y pocas combinaciones (depurar en minutos)
# "estandar" -> configuración recomendada para el reporte
# "completo" -> más datos y ~300 combinaciones (tarda bastante más)
CONFIGS = {
    "prueba": dict(
        n_muestra=6000,
        grid=dict(n_estimators=[5, 10, 20], max_depth=[3, 8, None],
                  max_features=["sqrt", 0.5], min_samples_split=[2, 10])),                      # 36 combinaciones
    "estandar": dict(
        n_muestra=30000,          # con más datos cada tarea tarda más y el arranque de procesos pesa menos
        grid=dict(n_estimators=[10, 25, 50, 100], max_depth=[5, 10, 20, None],
                  max_features=["sqrt", 0.4], min_samples_split=[2, 10],
                  min_samples_leaf=[1, 3])),                                                    # 128 combinaciones
    "completo": dict(
        n_muestra=100000,
        grid=dict(n_estimators=[10, 25, 50, 100, 150, 200], max_depth=[5, 10, 20, None],
                  max_features=["sqrt", 0.3, 0.5], min_samples_split=[2, 10],
                  min_samples_leaf=[1, 3])),                                                    # 288 combinaciones
}


def construir_tareas(grid_cfg, data_path, folds, seed):
    """Una tarea por combinación, en el orden de ParameterGrid."""
    grid = list(ParameterGrid(grid_cfg))
    return [(i, p, str(data_path), folds, seed) for i, p in enumerate(grid)]


def costo_estimado(params):
    """Heurística de costo relativo (sin ejecutar nada); sirve para ordenar tareas de más a menos pesada."""
    mf = params.get("max_features", "sqrt")
    mf = 0.13 if mf in ("sqrt", "log2") else float(mf)
    depth = params.get("max_depth") or 30
    leaf = params.get("min_samples_leaf", 1)
    split = params.get("min_samples_split", 2)
    return params["n_estimators"] * mf * (1 + math.log2(depth)) / (1 + 0.15 * (leaf - 1)) / (1 + 0.01 * (split - 2))


def asignar_worker(df):
    """Numera los trabajadores 0..p-1 según el orden en que empezaron a trabajar (a partir del PID)."""
    primero = df.groupby("pid")["t_start"].min().sort_values()
    mapa = {pid: i for i, pid in enumerate(primero.index)}
    df = df.copy()
    df["worker"] = df["pid"].map(mapa)
    return df


class Registro:
    """Guarda cada corrida en disco (CSV + manifiesto JSON) para poder retomar si se reinicia el kernel."""

    def __init__(self, carpeta):
        self.dir = Path(carpeta)
        self.manifest = self.dir / "runs.json"

    def cargar(self):
        return json.load(open(self.manifest)) if self.manifest.exists() else []

    def existe(self, name, p, rep):
        return any(r["name"] == name and r["p"] == p and r["rep"] == rep for r in self.cargar())

    def guardar(self, name, p, rep, wall, t0, df):
        self.dir.mkdir(parents=True, exist_ok=True)
        fn = self.dir / f"{name}_p{p}_r{rep}.csv"
        df.to_csv(fn, index=False)
        runs = [r for r in self.cargar() if not (r["name"] == name and r["p"] == p and r["rep"] == rep)]
        runs.append(dict(name=name, p=p, rep=rep, wall=wall, t0=t0, file=fn.name))
        json.dump(runs, open(self.manifest, "w"), indent=1)

    def todas(self):
        """Devuelve {(name, p, rep): dict(wall, t0, df)}."""
        return {(r["name"], r["p"], r["rep"]): dict(wall=r["wall"], t0=r["t0"], df=pd.read_csv(self.dir / r["file"]))
                for r in self.cargar()}
