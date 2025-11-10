#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "io_reader.h"
#include "matcher_api.h"

enum {
    TAG_PIC_HDR   = 1,
    TAG_PIC_DATA  = 2,
    TAG_RESULT    = 3,
    TAG_TERMINATE = 9
};

static const char* kInputPath  = "input.txt";
static const char* kOutputPath = "output.txt";

static void die_mpi(int rc, const char* where){
    if(rc != MPI_SUCCESS){
        fprintf(stderr, "MPI error at %s (rc=%d)\n", where, rc);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

static void bcast_problem_objects_and_threshold(Problem* prob, int rank_root, int my_rank){
    die_mpi(MPI_Bcast(&prob->threshold, 1, MPI_DOUBLE, rank_root, MPI_COMM_WORLD), "Bcast threshold");
    die_mpi(MPI_Bcast(&prob->num_objects, 1, MPI_INT, rank_root, MPI_COMM_WORLD), "Bcast num_objects");

    if (my_rank != rank_root){
        prob->objects = (Object*)calloc((size_t)prob->num_objects, sizeof(Object));
        if(!prob->objects){ perror("calloc objects"); MPI_Abort(MPI_COMM_WORLD, 2); }
    }

    for (int k=0; k<prob->num_objects; ++k){
        Object* O = &prob->objects[k];
        die_mpi(MPI_Bcast(&O->id, 1, MPI_INT, rank_root, MPI_COMM_WORLD), "Bcast obj id");
        die_mpi(MPI_Bcast(&O->M,  1, MPI_INT, rank_root, MPI_COMM_WORLD), "Bcast obj M");

        size_t mm = (size_t)O->M * (size_t)O->M;
        if (my_rank != rank_root){
            O->data = (int*)malloc(mm * sizeof(int));
            if(!O->data){ perror("malloc O->data"); MPI_Abort(MPI_COMM_WORLD, 2); }
        }
        die_mpi(MPI_Bcast(O->data, (int)mm, MPI_INT, rank_root, MPI_COMM_WORLD), "Bcast obj data");
    }
}

static ObjView* build_obj_views(const Problem* prob){
    int K = prob->num_objects;
    ObjView* v = (ObjView*)malloc((size_t)K * sizeof(ObjView));
    if(!v){ perror("malloc ObjView"); return NULL; }
    for(int k=0;k<K;k++){
        v[k].id   = prob->objects[k].id;
        v[k].M    = prob->objects[k].M;
        v[k].data = prob->objects[k].data;
    }
    return v;
}

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);

    int rank=0, world=1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    const int ROOT = 0;

    if(world < 2){
        if(rank==0) fprintf(stderr, "Need at least 2 MPI ranks\n");
        MPI_Finalize();
        return 1;
    }

    Problem prob; memset(&prob, 0, sizeof(prob));

    if(rank == ROOT){
        double t0 = MPI_Wtime();

        if(read_input_file_strict(kInputPath, &prob) != 0){
            fprintf(stderr, "Input parse error.\n");
            MPI_Abort(MPI_COMM_WORLD, 2);
        }

        FILE* outfp = fopen(kOutputPath, "w");
        if(!outfp){ perror("fopen output.txt"); MPI_Abort(MPI_COMM_WORLD, 2); }

        bcast_problem_objects_and_threshold(&prob, ROOT, rank);

        int next_pic = 0;
        int active   = world - 1;

        for(int p=1; p<world && next_pic < prob.num_pictures; ++p){
            const Picture* Pic = &prob.pictures[next_pic];
            int hdr[2] = { Pic->id, Pic->N };
            MPI_Send(hdr, 2, MPI_INT, p, TAG_PIC_HDR, MPI_COMM_WORLD);
            MPI_Send(Pic->data, Pic->N*Pic->N, MPI_INT, p, TAG_PIC_DATA, MPI_COMM_WORLD);
            next_pic++;
        }

        while(active > 0){
            int res[5] = {0,0,0,0,0};
            MPI_Status st;
            MPI_Recv(res, 5, MPI_INT, MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &st);

            if(res[0]){
                fprintf(outfp, "Picture %d found Object %d in Position(%d,%d)\n",
                        res[1], res[2], res[3], res[4]);
            }else{
                fprintf(outfp, "Picture %d No Objects were found\n", res[1]);
            }
            fflush(outfp);

            int w = st.MPI_SOURCE;
            if(next_pic < prob.num_pictures){
                const Picture* Pic = &prob.pictures[next_pic];
                int hdr[2] = { Pic->id, Pic->N };
                MPI_Send(hdr, 2, MPI_INT, w, TAG_PIC_HDR, MPI_COMM_WORLD);
                MPI_Send(Pic->data, Pic->N*Pic->N, MPI_INT, w, TAG_PIC_DATA, MPI_COMM_WORLD);
                next_pic++;
            }else{
                MPI_Send(NULL, 0, MPI_INT, w, TAG_TERMINATE, MPI_COMM_WORLD);
                active--;
            }
        }

        double t1 = MPI_Wtime();
        double total_time = t1 - t0;

        // Print total time only to stdout
        printf("TotalTime = %.6f seconds\n", total_time);

        fclose(outfp);
        free_problem(&prob);

    } else {
        bcast_problem_objects_and_threshold(&prob, ROOT, rank);
        ObjView* views = build_obj_views(&prob);
        if(!views){ MPI_Abort(MPI_COMM_WORLD, 2); }

        while(1){
            MPI_Status st;
            int hdr[2] = {0,0};
            MPI_Recv(hdr, 2, MPI_INT, ROOT, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
            if(st.MPI_TAG == TAG_TERMINATE) break;

            int pic_id = hdr[0];
            int N      = hdr[1];
            size_t nn  = (size_t)N * (size_t)N;
            int* P = (int*)malloc(nn * sizeof(int));
            if(!P){ perror("malloc P"); MPI_Abort(MPI_COMM_WORLD, 2); }
            MPI_Recv(P, (int)nn, MPI_INT, ROOT, TAG_PIC_DATA, MPI_COMM_WORLD, &st);

            MatchResult out = (MatchResult){0, pic_id, -1, -1, -1};
            (void)matcher_exists_per_object_omp(P, N, pic_id,
                                                views, prob.num_objects,
                                                prob.threshold, &out);

            int res[5] = { out.found, out.pic_id, out.obj_id, out.pos_i, out.pos_j };
            MPI_Send(res, 5, MPI_INT, ROOT, TAG_RESULT, MPI_COMM_WORLD);

            free(P);
        }

        for(int k=0;k<prob.num_objects;k++) free(prob.objects[k].data);
        free(prob.objects);
        free(views);
    }

    MPI_Finalize();
    return 0;
}
