// matcher_api.h
#pragma once
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int id;         // object id
    int M;          // object size MxM
    int *data;      // host data (M*M)
} ObjView;

typedef struct {
    int id;         // picture id
    int N;          // picture size NxN
    int *data;      // host data (N*N)
} PictureView;

typedef struct {
    int found;      // 0/1
    int pic_id;     // which picture
    int obj_id;     // which object matched
    int pos_i;      // top-left row of match
    int pos_j;      // top-left col of match
} MatchResult;

// ---------- OpenMP wrapper (called by each MPI worker) ----------
int matcher_exists_per_object_omp(
    const int* hP, int N, int pic_id,
    const ObjView* objs, int K,
    double threshold,
    MatchResult* out);

// ---------- CUDA per-object launcher (called by OMP threads) ----------
int matcher_exists_per_object_cuda(
    const int* hP, int N, int pic_id,
    const ObjView* obj,           // ONE object (id,M,data)
    double threshold,
    int gpu_id,                   // device selector (tid % num_gpus)
    MatchResult* out);

#ifdef __cplusplus
}
#endif
