"""
Versión 3: MEMORIA DISTRIBUIDA con mpi4py, patrón MAESTRO-TRABAJADOR con reparto bajo demanda.

  - Rango 0 (maestro): no entrena; administra la cola de tareas.
  - Rangos 1..p (trabajadores): avisan que están libres (y entregan su resultado); el maestro responde
    con la siguiente combinación o con una señal de paro.

Uso (lo lanza el notebook; también se puede correr a mano):
    mpiexec -n <p+1> python paralelo_mpi/mpi_hpo.py <tasks.pkl> <prefijo_salida>
"""
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "comun"))

import pandas as pd
from mpi4py import MPI
from worker import evaluate, init_worker

TAG_TASK, TAG_STOP = 1, 2


def main(tasks_pkl, out_prefix):
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    n_workers = size - 1

    comm.Barrier()                                   # todos arrancan juntos
    t0_epoch, t0 = time.time(), MPI.Wtime()

    if rank == 0:                                    # ------------------- MAESTRO -------------------
        tasks = pickle.load(open(tasks_pkl, "rb"))
        nxt, activos, resultados = 0, n_workers, []
        status = MPI.Status()
        while activos > 0:
            msg = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)   # "estoy libre" (+ resultado)
            src = status.Get_source()
            if msg is not None:
                resultados.append(msg)
            if nxt < len(tasks):
                comm.send(tasks[nxt], dest=src, tag=TAG_TASK)
                nxt += 1                                                            # siguiente tarea
            else:
                comm.send(None, dest=src, tag=TAG_STOP)
                activos -= 1                                                        # ya no hay trabajo
        wall = MPI.Wtime() - t0
        pd.DataFrame(resultados).to_csv(f"{out_prefix}.csv", index=False)
        json.dump(dict(wall=wall, t0=t0_epoch), open(f"{out_prefix}.json", "w"))
    else:                                            # ------------------- TRABAJADORES -------------------
        path = pickle.load(open(tasks_pkl, "rb"))[0][2]
        init_worker(path)                            # carga el dataset una sola vez
        comm.send(None, dest=0, tag=TAG_TASK)        # "estoy listo"
        while True:
            status = MPI.Status()
            tarea = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            if status.Get_tag() == TAG_STOP:
                break
            res = evaluate(tarea)
            res["rank"] = rank
            comm.send(res, dest=0, tag=TAG_TASK)     # devuelve resultado y pide más


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
