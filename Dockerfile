FROM python:3.11-slim

# OpenMPI + compilador (mpi4py se compila contra MPI)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential openmpi-bin libopenmpi-dev && \
    rm -rf /var/lib/apt/lists/*

# Dentro de un contenedor se corre como root y a veces hay menos "slots" que procesos
ENV OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    OMPI_MCA_rmaps_base_oversubscribe=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/app"]
