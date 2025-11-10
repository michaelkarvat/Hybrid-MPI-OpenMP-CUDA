#include "io_reader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <ctype.h>

/* ---------- local helpers ---------- */
static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) {
        fprintf(stderr, "malloc(%zu) failed: %s\n", n, strerror(errno));
        exit(1);
    }
    return p;
}

/* Read next non-empty line. Trims newline/CR and leading spaces. */
static int read_line(FILE *f, char *buf, size_t cap) {
    while (fgets(buf, cap, f)) {
        size_t n = strlen(buf);
        while (n > 0 && (buf[n-1] == '\n' || buf[n-1] == '\r')) buf[--n] = '\0';
        char *p = buf;
        while (*p && isspace((unsigned char)*p)) p++;
        if (*p == '\0') continue; /* blank line → skip */
        if (p != buf) memmove(buf, p, strlen(p)+1);
        return 1;
    }
    return 0;
}

static int parse_int_line(FILE *f, int *out) {
    char buf[256];
    if (!read_line(f, buf, sizeof(buf))) return 0;
    char *end = NULL;
    long v = strtol(buf, &end, 10);
    while (*end && isspace((unsigned char)*end)) end++;
    if (*end != '\0') {
        fprintf(stderr, "Expected single INT line, got: '%s'\n", buf);
        return 0;
    }
    *out = (int)v;
    return 1;
}

static int parse_double_line(FILE *f, double *out) {
    char buf[256];
    if (!read_line(f, buf, sizeof(buf))) return 0;
    char *end = NULL;
    double v = strtod(buf, &end);
    while (*end && isspace((unsigned char)*end)) end++;
    if (*end != '\0') {
        fprintf(stderr, "Expected single FLOAT line, got: '%s'\n", buf);
        return 0;
    }
    *out = v;
    return 1;
}

/* ---------- public API ---------- */
void free_problem(Problem *P) {
    if (!P) return;
    for (int i = 0; i < P->num_pictures; ++i) free(P->pictures[i].data);
    free(P->pictures);
    for (int k = 0; k < P->num_objects; ++k) free(P->objects[k].data);
    free(P->objects);
    memset(P, 0, sizeof(*P));
}

int read_input_file_strict(const char *path, Problem *out) {
    memset(out, 0, sizeof(*out));
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open '%s': %s\n", path, strerror(errno)); return 1; }

    /* threshold */
    if (!parse_double_line(f, &out->threshold)) { fprintf(stderr, "Missing/invalid threshold\n"); fclose(f); return 2; }

    /* pictures */
    if (!parse_int_line(f, &out->num_pictures)) { fprintf(stderr, "Missing/invalid number of pictures\n"); fclose(f); return 3; }
    if (out->num_pictures < 0) { fprintf(stderr, "num_pictures must be >= 0\n"); fclose(f); return 4; }
    out->pictures = (Picture*)xmalloc((size_t)out->num_pictures * sizeof(Picture));

    for (int p = 0; p < out->num_pictures; ++p) {
        Picture *P = &out->pictures[p];
        if (!parse_int_line(f, &P->id)) { fprintf(stderr, "Missing picture id at idx %d\n", p); fclose(f); return 5; }
        if (!parse_int_line(f, &P->N )) { fprintf(stderr, "Missing picture N at idx %d\n", p); fclose(f); return 6; }
        if (P->N <= 0)               { fprintf(stderr, "Picture N must be > 0 (id=%d)\n", P->id); fclose(f); return 7; }

        size_t nn = (size_t)P->N * (size_t)P->N;
        P->data = (int*)xmalloc(nn * sizeof(int));
        for (size_t i = 0; i < nn; ++i) {
            if (!parse_int_line(f, &P->data[i])) {
                fprintf(stderr, "Picture #%d: expected %zu element lines\n", P->id, nn);
                fclose(f); return 8;
            }
        }
    }

    /* objects */
    if (!parse_int_line(f, &out->num_objects)) { fprintf(stderr, "Missing/invalid number of objects\n"); fclose(f); return 9; }
    if (out->num_objects < 0) { fprintf(stderr, "num_objects must be >= 0\n"); fclose(f); return 10; }
    out->objects = (Object*)xmalloc((size_t)out->num_objects * sizeof(Object));

    for (int k = 0; k < out->num_objects; ++k) {
        Object *O = &out->objects[k];
        if (!parse_int_line(f, &O->id)) { fprintf(stderr, "Missing object id at idx %d\n", k); fclose(f); return 11; }
        if (!parse_int_line(f, &O->M )) { fprintf(stderr, "Missing object M at idx %d\n",  k); fclose(f); return 12; }
        if (O->M <= 0)               { fprintf(stderr, "Object M must be > 0 (id=%d)\n", O->id); fclose(f); return 13; }

        size_t mm = (size_t)O->M * (size_t)O->M;
        O->data = (int*)xmalloc(mm * sizeof(int));
        for (size_t i = 0; i < mm; ++i) {
            if (!parse_int_line(f, &O->data[i])) {
                fprintf(stderr, "Object #%d: expected %zu element lines\n", O->id, mm);
                fclose(f); return 14;
            }
        }
    }

    fclose(f);
    return 0;
}

void print_summary_details(const Problem *P) {
    printf("Matching value (threshold): %.6f\n", P->threshold);
    printf("Pictures: %d\n", P->num_pictures);
    for (int p = 0; p < P->num_pictures; ++p) {
        const Picture *Pic = &P->pictures[p];
        printf("  Picture #%d: id=%d, N=%d (matrix %dx%d)\n", p, Pic->id, Pic->N, Pic->N, Pic->N);
    }
    printf("Objects: %d\n", P->num_objects);
    for (int k = 0; k < P->num_objects; ++k) {
        const Object *Obj = &P->objects[k];
        printf("  Object  #%d: id=%d, M=%d (matrix %dx%d)\n", k, Obj->id, Obj->M, Obj->M, Obj->M);
    }
}

/* ---------- MPI broadcast helper (optional) ---------- */
#ifdef USE_MPI
int mpi_bcast_problem(Problem *P, int root, MPI_Comm comm) {
    int rank; MPI_Comm_rank(comm, &rank);

    /* Threshold */
    MPI_Bcast(&P->threshold, 1, MPI_DOUBLE, root, comm);

    /* Picture count */
    MPI_Bcast(&P->num_pictures, 1, MPI_INT, root, comm);
    if (rank != root) P->pictures = (P->num_pictures>0) ? (Picture*)xmalloc((size_t)P->num_pictures*sizeof(Picture)) : NULL;

    /* Each picture: id, N, then N*N ints */
    for (int p = 0; p < P->num_pictures; ++p) {
        MPI_Bcast(&P->pictures[p].id, 1, MPI_INT, root, comm);
        MPI_Bcast(&P->pictures[p].N , 1, MPI_INT, root, comm);
        int N = P->pictures[p].N;
        size_t nn = (size_t)N * (size_t)N;
        if (rank != root) P->pictures[p].data = (int*)xmalloc(nn*sizeof(int));
        MPI_Bcast(P->pictures[p].data, (int)nn, MPI_INT, root, comm);
    }

    /* Object count */
    MPI_Bcast(&P->num_objects, 1, MPI_INT, root, comm);
    if (rank != root) P->objects = (P->num_objects>0) ? (Object*)xmalloc((size_t)P->num_objects*sizeof(Object)) : NULL;

    /* Each object: id, M, then M*M ints */
    for (int k = 0; k < P->num_objects; ++k) {
        MPI_Bcast(&P->objects[k].id, 1, MPI_INT, root, comm);
        MPI_Bcast(&P->objects[k].M , 1, MPI_INT, root, comm);
        int M = P->objects[k].M;
        size_t mm = (size_t)M * (size_t)M;
        if (rank != root) P->objects[k].data = (int*)xmalloc(mm*sizeof(int));
        MPI_Bcast(P->objects[k].data, (int)mm, MPI_INT, root, comm);
    }
    return 0;
}
#endif
