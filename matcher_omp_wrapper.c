// matcher_omp_wrapper.c  — OpenMP wrapper using only critical (no C11 atomics)
// Build:  gcc -O3 -fopenmp -std=c11 -Wall -Wextra -c matcher_omp_wrapper.c

#include <omp.h>
#include <string.h>   // memset

// Project header: adjust to your repo. Must declare ObjView, MatchResult,
// and matcher_exists_per_object_cuda(...).
#include "matcher_api.h"

int matcher_exists_per_object_omp(
    const int* hP, int N, int pic_id,
    const ObjView* objs, int K,
    double threshold,
    MatchResult* out)
{
    if (!hP || !objs || K < 0 || !out) return -1;

    // Default "not found"
    memset(out, 0, sizeof(*out));
    if (K == 0) return 0;

    // Shared state guarded by critical sections
    int found = 0;                // 0 = no winner; 1 = winner chosen
    MatchResult winner = *out;    // published winner

    #pragma omp parallel for schedule(static) default(none) \
            shared(found, winner, objs, K, hP, N, pic_id, threshold, out)
    for (int k = 0; k < K; ++k) {
        // Check early-exit flag under the same named critical to avoid data races
        int skip_this = 0;
        #pragma omp critical(flag)
        { skip_this = found; }
        if (skip_this) continue;

        const ObjView* obj = &objs[k];

        // Use thread id as gpu_id (your CUDA code can map/mod to actual device)
        int tid = omp_get_thread_num();

        MatchResult local;
        memset(&local, 0, sizeof(local));

        int rc = matcher_exists_per_object_cuda(
                     hP, N, pic_id,
                     obj, threshold,
                     /*gpu_id*/ tid,
                     &local);

        if (rc == 0 && local.found) {
            // Publish the first winner exactly once:
            // re-check and set 'found' while holding the same lock.
            #pragma omp critical(flag)
            {
                if (!found) {
                    winner = local;
                    found  = 1;
                }
            }
        }
    } // implicit barrier here

    if (found) {
        *out = winner;
    }
    return 0;
}
