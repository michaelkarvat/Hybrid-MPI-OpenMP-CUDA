#ifndef IO_READER_H
#define IO_READER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int id, N;      /* N x N */
    int *data;      /* length N*N */
} Picture;

typedef struct {
    int id, M;      /* M x M */
    int *data;      /* length M*M */
} Object;

typedef struct {
    double threshold;
    int num_pictures; Picture *pictures;
    int num_objects;  Object  *objects;
} Problem;

/* Read from disk (strict: exactly one number per line for every element).
   Returns 0 on success, non-zero on error. */
int read_input_file_strict(const char *path, Problem *out);

/* Free all allocations inside Problem (safe to call once). */
void free_problem(Problem *P);

/* Print only IDs and sizes (no full matrices). */
void print_summary_details(const Problem *P);

#ifdef USE_MPI
#include <mpi.h>
/* Broadcast the fully populated Problem from 'root' to all ranks in 'comm'.
   On non-root ranks, allocations are performed by this function.
   Returns 0 on success, non-zero on error. */
int mpi_bcast_problem(Problem *P, int root, MPI_Comm comm);
#endif

#ifdef __cplusplus
}
#endif
#endif /* IO_READER_H */
