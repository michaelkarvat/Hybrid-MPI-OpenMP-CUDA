# Hybrid MPI + OpenMP + CUDA Matcher

A high-performance computing project implementing a hybrid parallel architecture using MPI, OpenMP, and CUDA for accelerated pattern matching, executed on an HPC cluster using SLURM.

## Overview

This project demonstrates a hybrid parallel computing solution that combines distributed computing, CPU multithreading, and GPU acceleration to process large workloads efficiently.

The system distributes tasks between processes, parallelizes computation on CPU cores, and offloads intensive operations to the GPU.

## Architecture

The application follows a hybrid execution model:

1. **MPI** distributes input data and coordinates work across multiple processes.
2. **OpenMP** parallelizes execution across CPU cores inside each process.
3. **CUDA** accelerates compute-heavy operations using GPU kernels.
4. **SLURM** manages job scheduling and resource allocation on the cluster.

## Project Structure

- `mpi_driver.c` – main entry point and distributed execution controller  
- `matcher_omp_wrapper.c` – CPU multithreading layer using OpenMP  
- `matcher_cuda_per_object.cu` – CUDA kernel for GPU acceleration  
- `io_reader.c / io_reader.h` – input/output handling  
- `matcher_api.h` – matching interface  
- `input.txt` – sample input  
- `output.txt` – generated output  

## Technologies

- C  
- CUDA  
- MPI  
- OpenMP  
- SLURM  
- Parallel algorithm design  

## Execution Environment

The project was executed on an HPC cluster using the SLURM workload manager.

Capabilities demonstrated:

- Distributed execution (MPI)
- Multi-core CPU parallelism (OpenMP)
- GPU acceleration (CUDA)
- Cluster job scheduling (SLURM)

## Build

```bash
- sbatch finalfromwsl 
