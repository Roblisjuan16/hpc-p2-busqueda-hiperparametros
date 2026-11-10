# Búsqueda distribuida de hiperparámetros (Proyecto P2) — Cómputo de Alto Rendimiento

Evidencia de producto P2 · Ingeniería en Datos e Inteligencia Artificial, 7.º cuatrimestre.

Se busca la mejor combinación de hiperparámetros de un **Random Forest** sobre el dataset **Covertype (UCI)** evaluando
cientos de combinaciones con validación cruzada, y se compara el rendimiento de:

| Versión | Carpeta | Modelo de paralelismo |
|---|---|---|
| Secuencial (línea base, ciclo `for`) | `secuencial/` | — |
| `multiprocessing.Pool` (reparto estático / dinámico / dinámico ordenado) y `joblib` | `paralelo_pool/` | procesos locales (memoria compartida) |
| `mpi4py`, patrón maestro–trabajador | `paralelo_mpi/` | memoria distribuida (paso de mensajes) |
| Ray (opcional) | `paralelo_ray/` | memoria distribuida |

Se miden tiempo, *speedup* `S(p)=T1/Tp`, eficiencia `E(p)=S(p)/p`, desbalance de carga y la fracción paralela estimada con la
Ley de Amdahl, con 1, 2, 4 y 8 procesos y varias repeticiones (media ± desviación estándar).

## Estructura

```
├── comun/            worker.py (función de trabajo compartida) · utils_hpo.py (configuración, tareas, registro)
├── secuencial/       secuencial.py
├── paralelo_pool/    pool_hpo.py
├── paralelo_mpi/     mpi_hpo.py
├── paralelo_ray/     ray_hpo.py
├── experimentos/     HPC_P2_Busqueda_Hiperparametros.ipynb   <- informe + ejecución de todos los experimentos
├── resultados/       (se crea al ejecutar) CSV por corrida, resumen_metricas.csv, amdahl.csv y figuras .png
├── datos/            (se crea al ejecutar) caché del dataset
├── requirements.txt · Dockerfile
```

## Instalación

### Opción A — entorno local (Linux / macOS / Windows)

1. Python 3.10 o superior.
2. Una implementación de MPI (necesaria para `mpi4py`):
   - Ubuntu/Debian: `sudo apt install openmpi-bin libopenmpi-dev`
   - macOS: `brew install open-mpi`
   - Windows: instalar [MS-MPI](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi) (Runtime y SDK).
3. Dependencias de Python:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # en Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Opción B — Docker

```bash
docker build -t hpc-p2 .
docker run --rm -p 8888:8888 -v "$(pwd)/resultados:/app/resultados" hpc-p2
```

Abre en el navegador la URL con token que imprime JupyterLab. El contenedor usa los núcleos que Docker le asigne
(en Docker Desktop se ajustan en *Settings → Resources*).

## Cómo ejecutar todo

1. Abrir `experimentos/HPC_P2_Busqueda_Hiperparametros.ipynb` **desde la carpeta `experimentos/`** (para que encuentre los módulos).
2. Ajustar la celda de configuración (sección 3):
   - `MODO = "prueba"` para verificar que todo funciona (minutos); `"estandar"` para el reporte; `"completo"` para más datos y ~288 combinaciones.
   - `REPS` (repeticiones por configuración; el enunciado pide al menos 3) y `MAX_PROCS` (1, 2, 4, 8 según los núcleos).
   - `USAR_DATOS_SINTETICOS = False` para usar el Covertype real (se descarga solo la primera vez; requiere internet).
3. *Run All*. Cada corrida se guarda en `resultados/`; si se interrumpe, al volver a ejecutar se omiten las ya hechas (`REHACER = False`).

> El tiempo total depende del equipo: el notebook imprime una estimación después de medir la versión secuencial.
> Para ir más rápido: bajar `REPS`, usar `MODO = "prueba"` o reducir `n_muestra` en `comun/utils_hpo.py`.

### Ejecutar solo MPI a mano (opcional)

```bash
mpiexec -n 5 python paralelo_mpi/mpi_hpo.py resultados/tasks.pkl resultados/mpi_manual   # 1 maestro + 4 trabajadores
```
(`tasks.pkl` lo genera el notebook.) Si `mpiexec` se queja de que no hay suficientes *slots*, añadir `--oversubscribe` (OpenMPI).

## Reproducibilidad

- Semillas fijas (`SEMILLA = 42`) en el muestreo, la partición, la validación cruzada y el Random Forest: cada combinación
  produce siempre el mismo puntaje, sin importar la versión; el notebook lo verifica (sección 6.2).
- Los tiempos dependen del equipo; las especificaciones (CPU, núcleos, RAM, GPU) se imprimen al inicio del notebook.
- Para comparar con la misma carga, cerrar otros programas pesados durante las mediciones.

## Declaración de uso de IA

Ver la sección 9 del notebook.
