# Hybrid MPI + OpenMP + CUDA – Parallel Submatrix Search

A high-performance computing project implementing a hybrid parallel solution (MPI + OpenMP + CUDA) for solving the Submatrix Search problem on an HPC cluster using SLURM.

## Overview
This project detects whether a smaller matrix ("Object") appears inside a larger matrix ("Picture") using distributed computing, CPU multithreading, and GPU acceleration. The system combines MPI for node communication, OpenMP for CPU parallelism, CUDA for GPU computation, and SLURM for job scheduling.

## Problem Definition
For each possible position (I,J) of the Object inside the Picture, a matching score is calculated:

diff = abs((p - o) / p)

Matching(I,J) = Σ abs((p - o) / p)

A match is detected if the matching score is below a predefined threshold.

## Example

Picture (6x6):

10  5  67 12  8  4  
23  6   5 14  9  5  
12 10  20 56  2  3  
1   2   6 10  3  2  
45  3   7  5  5  2  
11 43   2 54  1 12  

Object (3x3):

5 14  9  
20 56 2  
6 10  3  

The algorithm scans the Picture and computes a matching score for each possible position of the Object inside the Picture.  
If the matching value is below the defined threshold → a match is detected.

## Input
`input.txt` contains:
- Matching threshold value
- Number of Pictures
- For each Picture: ID, size (N), NxN matrix values
- Number of Objects
- For each Object: ID, size, matrix values

Input is initially loaded by the master process and distributed across MPI processes.

## Output
`output.txt` format:

Picture <ID> found Object <ID> in Position (I,J)

or

Picture <ID> No Objects were found

Results are aggregated and written by the master MPI process.

## Parallel Architecture
- **MPI** – distributes workload across nodes
- **OpenMP** – parallel CPU execution per process
- **CUDA** – GPU acceleration of matching computations
- **SLURM** – resource allocation and job scheduling

## Execution Flow
1. Master process loads input data
2. MPI distributes Pictures and Objects between nodes
3. Each process scans its assigned workload
4. OpenMP parallelizes CPU computation
5. CUDA accelerates matching calculations on GPU
6. Results are aggregated
7. Output is written to file

## Project Structure
- `mpi_driver.c` – distributed execution controller  
- `matcher_omp_wrapper.c` – OpenMP CPU parallelization  
- `matcher_cuda_per_object.cu` – CUDA kernel implementation  
- `io_reader.c / io_reader.h` – I/O handling  
- `matcher_api.h` – matching interface  
- `input.txt` – sample input  
- `output.txt` – results  
- `finalfromwsl` – SLURM batch script  

## Technologies
C, CUDA, MPI, OpenMP, SLURM, Parallel Algorithms, GPU Computing

## Execution Environment
HPC cluster with:
- Multiple compute nodes
- 8 CPU cores per task
- NVIDIA RTX GPU
- SLURM workload manager

Capabilities demonstrated:
- Distributed execution (MPI)
- Multi-core CPU parallelism (OpenMP)
- GPU acceleration (CUDA)
- Cluster scheduling and orchestration

## Build & Run

Run using SLURM:

```bash
sbatch finalfromwsl
