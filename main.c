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
#define TAG 0  // Tag for MPI messages.

int main(int argc, char* argv[]) {
    int rows,  // Rows in V matrix.
        cols, // Columns in V matrix.
        n_comp,  // Number of NMF components.
        n_iter,  // Number of iterations of the NMF algorithm.
        rank,  // ID of current task.
        size,  // Total number of tasks.
        bs_cols,  // Block size, number of columns of V stored on each process.
        bs_rows,  // Block size, number of rows of V stored on each process.
        numworkers,
        errorcode,
        iter,
        i, j, k;  // Index variables.

    double *Vcol,  // Subset of columns of V.
           *Vrow,  // Subset of rows of V.
           *Wmat,  // Dictionary, W.
           *Hmat,  // Activation, H.
           *WTV,
           *WTVblock,
           *WTW,
           *WTWH,
           *VHT,
           *VHTblock,
           *HHT,
           *WHHT,
           res2,  // Sum of squared residuals (each worker's contribution).
           err;  // Error, err = 0.5 * sqrt(sum(res2)).

    double randomDouble();  // Generates a random double between 0 and 1.

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
    n_iter = atoi(argv[4]);

    numworkers = size - 1;
    // TODO: account for when cols and/or rows does not split evenly among workers.
    // At the moment, assuming cols % numworkers = 0 and rows % numworkers = 0.
    bs_cols = (int) ((float) cols / numworkers);
    bs_rows = (int) ((float) rows / numworkers);

    Wmat = malloc(rows * n_comp * sizeof(double));
    Hmat = malloc(n_comp * cols * sizeof(double));

    // Allocate matrices on master.
    if(rank == MASTER) {
        WTV = malloc(n_comp * cols * sizeof(double));
        WTW = malloc(n_comp * n_comp * sizeof(double));
        WTWH = malloc(n_comp * cols * sizeof(double));
        VHT = malloc(rows * n_comp * sizeof(double));
        HHT = malloc(n_comp * n_comp * sizeof(double));
        WHHT = malloc(rows * n_comp * sizeof(double));

        // Randomly initialize Wmat and Hmat.
        srand(time(NULL));  // Seed for random number generator.

        for(i = 0; i < rows; i++) {
            for(k = 0; k < n_comp; k++) {
                Wmat[i * n_comp + k] = randomDouble();
            }
        }

        for(i = 0; i < cols; i++) {
            for(k = 0; k < n_comp; k++) {
                Hmat[k * cols + i] = randomDouble();
            }
        }

        
        // Send Wmat and Hmat to workers.
        // TODO: Non-blocking send. Send matrices to all workers, and use MPI_Waitall to wait for all of the workers to finish receiving the matrices.
        for(i = 1; i <= numworkers; i++) {
            MPI_Send(Wmat, rows * n_comp, MPI_DOUBLE, i, TAG, MPI_COMM_WORLD);
            MPI_Send(Hmat, n_comp * cols, MPI_DOUBLE, i, TAG, MPI_COMM_WORLD);
        }
    }
    // Allocate matrices on workers.
    else {
        Vcol = malloc(rows * bs_cols * sizeof(double));
        Vrow = malloc(bs_rows * cols * sizeof(double));
        WTVblock = malloc(n_comp * bs_cols * sizeof(double));
        VHTblock = malloc(bs_rows * n_comp * sizeof(double));

        // Initialize Vcol and Vrow.
        // In reality, Vcol and Vrow would be read from hard drives on the cluster, or something along those lines.
        for(i = 0; i < rows; i++) {
            for(j = 0; j < bs_cols; j++) {
                Vcol[i * bs_cols + j] = 0.0;
            }
        }
        for(i = 0; i < bs_rows; i++) {
            for(j = 0; j < cols; j++) {
                Vrow[i * cols + j] = 0.0;
            }
        }

        // Receive Wmat and Hmat from master.
        // TODO: Non-blocking receive. Use MPI_Waitall.
        MPI_Recv(Wmat, rows * n_comp, MPI_DOUBLE, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(Hmat, n_comp * cols, MPI_DOUBLE, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    }

    // Main algorithm loop.
    for(iter = 0; iter < n_iter; iter++) {

        /* Update H. */

        // Compute WTVblock.
        if(rank != MASTER) {
            for(k = 0; k < n_comp; k++) {
                for(j = 0; j < bs_cols; j++) {
                    WTVblock[k * bs_cols + j] = 0.0;  // Initialize element (k, j) to zero.
                    for(i = 0; i < rows; i++) {
                        // Note that the transpose of Wmat is accessed here.
                        WTVblock[k * bs_cols + j] += Wmat[i * rows + k] * Vcol[i * bs_cols + j];
                    }
                }
            }

            // Send WTVblock to master.
            MPI_Send(WTVblock, n_comp * bs_cols, MPI_DOUBLE, 0, TAG, MPI_COMM_WORLD);
        }
        // Receive WTVblock from workers, store corresponding columns of WTV.
        else {
            for(i = 1; i <= numworkers; i++) {
                MPI_Recv(WTVblock, n_comp * bs_cols, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

                // Store WTVblock in WTV.
                // TODO: make sure this is correct. These indeces are confusing.
                for(j = 0; j < bs_cols; j++) {
                    for(k = 0; k < n_comp; k++) {
                        WTV[k * cols + (rank - 1) * bs_cols + j] = WTVblock[k * bs_cols + j];
                    }
                }
            }

            // Compute WTW = W' * W
            for(k = 0; k < n_comp; k++) {
                for(i = 0; i < n_comp; i++) {
                    WTW[k * n_comp + i] = 0.0;  // Initialize element (k, i) to zero.
                    for(j = 0; j < rows; j++) {
                        // Note that the transpose of Wmat is accessed here (W' * W).
                        WTW[k * n_comp + i] += Wmat[j * rows + k] * Wmat[j * n_comp + i];
                    }
                }
            }

            // Compute WTWH = WTW * H
            for(k = 0; k < n_comp; k++) {
                for(j = 0; j < cols; j++) {
                    WTWH[k * cols + j] = 0.0;  // Initialize element (k, j) to zero.
                    for(i = 0; i < n_comp; i++) {
                        // Note that the transpose of Wmat is accessed here (W' * W).
                        WTWH[k * cols + j] += WTW[k * n_comp + i] * Hmat[i * cols + j];
                    }
                }
            }

            // Update H.
            for(i = 0; i < n_comp * cols; i++) {
                Hmat[i] = Hmat[i] * (WTV[i] / WTWH[i]);
            }
        }

        // TODO: Master send H to workers.
        // TODO: Workers receive H from master.

        // TODO:
        /* Update W. */
        // Similar to updating H.

        // TODO: Master send W to workers.
        // TODO: Workers receive W from master.

        // Compute the sum of squared residuals.
        // TODO: don't compute the error more often than necessary. It requires too much communication.
        if(rank != MASTER) {
            res2 = 0.0;
            for(i = 0; i < rows; i++) {
                for(j = 0; j < bs_cols; j++) {
                    for(k = 0; k < n_comp; k++) {
                        // Squared difference between V and its reconstruction at point (i, j).
                        res2 += pow(Vcol[i * bs_cols + j] - Wmat[i * n_comp + k] * Hmat[k * cols + j], 2);
                    }
                }
            }

            // Send res2 to master.
            MPI_Send(&res2, 1, MPI_DOUBLE, MASTER, TAG, MPI_COMM_WORLD);
        }
        // 
        else {
            // Receive res2 from workers, sum into err.
            err = 0.0;
            for(i = 1; i <= numworkers; i++) {
                MPI_Recv(&res2, 1, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                err += res2;
            }

            // Compute reconstruction error from sum of squared residuals.
            err = 0.5 * sqrt(err);
            printf("Reconstruction error: %.4e\n", err);
        }
    }


    // Free memory allocated for all matrices.
    // TODO: free momory of all matrices.
    free(Wmat);
    free(Hmat);

    MPI_Finalize();

    return 0;
}

double randomDouble() {
    double r = (double) rand()/(double) RAND_MAX;
    return r;
}

