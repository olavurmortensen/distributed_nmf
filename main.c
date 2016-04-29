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
        n_iter,  // Number of iterations of the NMF algorithm.
        rank,  // ID of current task.
        size,  // Total number of tasks.
        tag = 0,  // Tag for MPI messages.
        bs_cols,  // Block size, number of columns of V stored on each process.
        bs_rows,  // Block size, number of rows of V stored on each process.
        numworkers,
        errorcode,
        iter,
        i, j, k;  // Index variables.

    double *Vcol,  // Subset of columns of V.
           *Vrow,  // Subset of rows of V.
           *Vrec,  // Reconstruction of V. Vrec = W * H. 
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
           res2_buf,  // res2 buffer.
           err;  // Error, err = 0.5 * sqrt(sum(res2)).

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

        // Set all matrices above to zero.

        // Randomly initialize Wmat and Hmat.
        
        // Send Wmat and Hmat to workers.
    }
    // Allocate matrices on workers.
    else {
        Vcol = malloc(rows * bs_cols * sizeof(double));
        Vrow = malloc(bs_rows * cols * sizeof(double));
        WTVblock = malloc(n_comp * bs_cols * sizeof(double));
        VHTblock = malloc(bs_rows * n_comp * sizeof(double));
        Vrec = malloc(rows * bs_cols * sizeof(double));

        // Initialize Vcol and Vrow.
        // In reality, Vcol and Vrow would be read from hard drives on the cluster.
    }

    // Main algorithm loop.
    for(iter = 0; iter < n_iter; iter++) {

        /* Update H. */

        // Compute WTVblock.
        if(rank != MASTER) {
            for(k = 0; k < n_comp; k++) {
                for(i = 0; i < rows; i++) {
                    for(j = 0; j < bs_cols; j++) {
                        // Note that the transpose of Wmat is accessed here.
                        // TODO: make sure this is correct. These indeces are confusing.
                        WTVblock[k * bs_cols + j] += Wmat[i * rows + k] * Vcol[i * bs_cols + j];
                    }
                }
            }

            // Send WTVblock to master.
            MPI_Send(WTVblock, n_comp * bs_cols, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
        }
        // Receive WTVblock from workers, store corresponding columns of WTV.
        else {
            for(i = 1; i < size; i++) {
                MPI_Recv(WTVblock, n_comp * bs_cols, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

                // TODO: Store WTVblock in WTV.
            }

            // Compute WTW = W' * W
            // First, we set WTW to zero.
            for(i = 0; i < n_comp * n_comp; i++) {
                WTW[i] = 0;
            }
            for(k = 0; k < n_comp; k++) {
                for(i = 0; i < n_comp; i++) {
                    for(j = 0; j < rows; j++) {
                        // Note that the transpose of Wmat is accessed here (W' * W).
                        WTW[k * n_comp + i] += Wmat[j * rows + k] * Wmat[j * n_comp + i];
                    }
                }
            }

            // Compute WTWH = WTW * H
            for(i = 0; i < n_comp * cols; i++) {
                WTWH[i] = 0;
            }
            for(k = 0; k < n_comp; k++) {
                for(i = 0; i < n_comp; i++) {
                    for(j = 0; j < cols; j++) {
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

        /* Update W. */
        // Similar to updating H.

        // Compute Vrec.
        if(rank != MASTER) {
            for(i = 0; i < rows; i++) {
                for(j = 0; j < bs_cols; j++) {
                    Vrec[i * bs_cols + j] = 0.0;  // Initialize element (i, j) to zero.
                    for(k = 0; k < n_comp; k++) {
                        Vrec[i * bs_cols + j] += Wmat[i * n_comp + k] * Hmat[k * cols + j];
                    }
                }
            }

            // Compute the sum of squared residuals for Vcol.
            res2 = 0.0;
            for(i = 0; i < rows * bs_cols; i++) {
                res2 += pow(Vcol[i] - Vrec[i], 2);
            }

            // Send res2 to master.
        }
        // 
        else {
            // Receive res2 from workers, sum into err.
            err = 0.0;
            for(i = 0; i < numworkers; i++) {
                MPI_Recv(&res2_buf, 1, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                err += res2_buf;
            }

            // Compute reconstruction error from sum of squared residuals.
            err = 0.5 * sqrt(err);
            printf("Reconstruction error: %.4e\n", err);

        }


    }


    // Free memory allocated for all matrices.
    free(Wmat);
    free(Hmat);

    MPI_Finalize();

    return 0;
}

