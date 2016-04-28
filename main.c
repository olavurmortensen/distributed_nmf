/*
 * FILE: main.c
 *
 * DESCRIPTION:
 *
 * Compute Non-negative Matrix Factorization (NMF) of an input matrix V into dictionary W and activation H.
 *
 * Input:
 * 
 * Call example:
 *
 * AUTHOR: Olavur Mortensen.
 * 
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MASTER 0  // Rank of master task.

int main(int argc, char* argv[]) {
    int rows,  // Rows in V matrix.
        cols, // Columns in V matrix.
        n_comp,  // Number of NMF components.
        rank,  // ID of current task.
        size,  // Total number of tasks.
        bsize,  // Block size.
        numworkers,
        errorcode,
        i, j, k;  // Index variables.

    double *Vmat,  // Block of matrix A of size (bsize x Acols).
           *Wmat,  // Block of matrix B of size (Brows x bsize).
           *Hmat;  // Block of matrix C of size (bsize x bsize).

    MPI_Status status;

    // MPI initialization.
    errorcode = MPI_Init(&argc, &argv);
    if (errorcode != MPI_SUCCESS) {
        printf("Error starting MPI program. Terminating.\n");
        MPI_Abort(MPI_COMM_WORLD, errorcode);
    }

    // Get process id (rank) and total number of processes (size).
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(size < 2) {
        printf("dNMF needs at least 2 MPI processes, quitting.\n");
        MPI_Abort(MPI_COMM_WORLD, errorcode);
        exit(1);
    }

    rows = atoi(argv[1]);
    cols = atoi(argv[2]);
    n_comp = atoi(argv[3]);

    numworkers = size - 1;
    bsize = (int) ((float) cols / numworkers);
    printf("%d\n", bsize);

    if(rank == MASTER) {
        Wmat = malloc(cols * n_comp * sizeof(double));

        free(Wmat);
    }

    MPI_Finalize();

    return 0;
}

