/*
 * FILE: main.c
 *
 * DESCRIPTION:
 *
 * Compute Non-negative Matrix Factorization (NMF) of an input matrix V into dictionary W and activation H.
 *
 * Input:
 * rows:        Number of rows in V.
 * cols:        Number of columns in V.
 * n_comp:      Number of components, common dimension of W and H.
 * n_iter:      Number of iterations of the optimization algorithm.
 * 
 * Call example:
 * mpiexec -n 100 ./dnmf_exec 1000 1000 10 100
 *
 * AUTHOR: Olavur Mortensen.
 * 
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>

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
        i, j, k, m;  // Index variables.

    double *Vcol,  // Subset of columns of V.
           *Vrow,  // Subset of rows of V.
           *Wmat_worker,  // Dictionary, W.
           *Hmat_worker,  // Activation, H.
           *Wmat_master,
           *Hmat_master,
           *WTV,
           *WTVblock_worker,
           *WTVblock_master,
           *WTW,
           *WTWH,
           *VHT,
           *VHTblock_worker,
           *VHTblock_master,
           *HHT,
           *WHHT,
           res2,  // Sum of squared residuals (each worker's contribution).
           res2_buff,
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

    /* Allocate matrices. */
    /* ----------------------------------------------- */
    if(rank == MASTER) {
        Wmat_master = malloc(rows * n_comp * sizeof(double));
        Hmat_master = malloc(n_comp * cols * sizeof(double));
        Wmat_worker = malloc(rows * n_comp * sizeof(double));
        Hmat_worker = malloc(n_comp * cols * sizeof(double));
        WTV = malloc(n_comp * cols * sizeof(double));
        WTW = malloc(n_comp * n_comp * sizeof(double));
        WTWH = malloc(n_comp * cols * sizeof(double));
        VHT = malloc(rows * n_comp * sizeof(double));
        HHT = malloc(n_comp * n_comp * sizeof(double));
        WHHT = malloc(rows * n_comp * sizeof(double));
        WTVblock_master = malloc(n_comp * bs_cols * sizeof(double));
        VHTblock_master = malloc(bs_rows * n_comp * sizeof(double));

        // Randomly initialize Wmat and Hmat.
        srand(time(NULL));  // Seed for random number generator.
        
        // TODO: add a small number ("epsilon") to W and H to ensure non-zero initial values.
        for(i = 0; i < rows; i++) {
            for(k = 0; k < n_comp; k++) {
                Wmat_master[i * n_comp + k] = randomDouble();
            }
        }

        for(i = 0; i < cols; i++) {
            for(k = 0; k < n_comp; k++) {
                Hmat_master[k * cols + i] = randomDouble();
            }
        }
    }
    else {
        // Allocate matrices on workers.
        Wmat_worker = malloc(rows * n_comp * sizeof(double));
        Hmat_worker = malloc(n_comp * cols * sizeof(double));
        Wmat_master = malloc(rows * n_comp * sizeof(double));
        Hmat_master = malloc(n_comp * cols * sizeof(double));
        Vcol = malloc(rows * bs_cols * sizeof(double));
        Vrow = malloc(bs_rows * cols * sizeof(double));
        WTVblock_worker = malloc(n_comp * bs_cols * sizeof(double));
        VHTblock_worker = malloc(bs_rows * n_comp * sizeof(double));

        // Initialize Vcol and Vrow.
        // In reality, Vcol and Vrow would be read from hard drives on the cluster, or something along those lines.
        // TODO: randomly initialize Vcol and communicate to get Vrow.
        for(i = 0; i < rows; i++) {
            for(j = 0; j < bs_cols; j++) {
                Vcol[i * bs_cols + j] = 1.0;
            }
        }
        for(i = 0; i < bs_rows; i++) {
            for(j = 0; j < cols; j++) {
                Vrow[i * cols + j] = 1.0;
            }
        }
    }

    /* Send W and H from master to workers using collective communications. */
    //printf("%d %d\n", errorcode, rank);
    MPI_Scatter(Wmat_master, rows * n_comp, MPI_DOUBLE, Wmat_worker, rows * n_comp, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    MPI_Scatter(Hmat_master, n_comp * cols, MPI_DOUBLE, Hmat_worker, n_comp * cols, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    if(rank == 19) {

        for(i = 0; i < cols; i++) {
            for(j = 0; j < n_comp; j++) {
                printf("%f ", Hmat_worker[i * n_comp + j]);
            }
            printf("\n");
        }
        printf("\n");
        printf("\n");

    }

    /* Main algorithm loop. */
    for(iter = 0; iter < n_iter; iter++) {
        /* Compute the sum of squared residuals. */
        /* -------------------------------------------------- */

        // TODO: don't compute the error more often than necessary. It requires too much communication.
        if(rank != MASTER) {
            res2 = 0.0;
            for(i = 0; i < rows; i++) {
                for(j = 0; j < bs_cols; j++) {
                    for(k = 0; k < n_comp; k++) {
                        // Squared difference between V and its reconstruction at point (i, j).
                        res2 += pow(Vcol[i * bs_cols + j] - Wmat_worker[i * n_comp + k] * Hmat_worker[k * cols + j], 2);
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
                MPI_Recv(&res2_buff, 1, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                err += res2_buff;
            }

            // Compute reconstruction error from sum of squared residuals.
            err = 0.5 * sqrt(err);
            printf("Reconstruction error: %.4e\n", err);
        }

        /* Update H. */
        /* -------------------------------------------------- */

        // Compute WTVblock.
        if(rank != MASTER) {
            for(k = 0; k < n_comp; k++) {
                for(j = 0; j < bs_cols; j++) {
                    WTVblock_worker[k * bs_cols + j] = 0.0;  // Initialize element (k, j) to zero.
                    for(i = 0; i < rows; i++) {
                        // Note that the transpose of Wmat is accessed here.
                        WTVblock_worker[k * bs_cols + j] += Wmat_worker[k * rows + i] * Vcol[i * bs_cols + j];
                    }
                }
            }

            // Send WTVblock to master.
            MPI_Send(WTVblock_worker, n_comp * bs_cols, MPI_DOUBLE, MASTER, TAG, MPI_COMM_WORLD);
        }
        // Receive WTVblock from workers, store corresponding columns of WTV.
        else {

            for(i = 1; i <= numworkers; i++) {
                MPI_Recv(WTVblock_master, n_comp * bs_cols, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

                // Store WTVblock in WTV.
                // TODO: make sure this is correct. These indeces are confusing.
                for(j = 0; j < bs_cols; j++) {
                    for(k = 0; k < n_comp; k++) {
                        WTV[k * cols + (i - 1) * bs_cols + j] = WTVblock_master[k * bs_cols + j];
                        //printf("%f\n", WTV[k * cols + (rank - 1) * bs_cols + j]);
                    }
                }
            }

            // Compute WTW = W' * W
            for(k = 0; k < n_comp; k++) {
                for(i = 0; i < n_comp; i++) {
                    WTW[k * n_comp + i] = 0.0;  // Initialize element (k, i) to zero.
                    for(j = 0; j < rows; j++) {
                        // Note that the transpose of Wmat is accessed here (W' * W).
                        WTW[k * n_comp + i] += Wmat_master[k * rows + i] * Wmat_master[j * n_comp + i];
                    }
                }
            }

            // Compute WTWH = WTW * H
            for(k = 0; k < n_comp; k++) {
                for(j = 0; j < cols; j++) {
                    WTWH[k * cols + j] = 0.0;  // Initialize element (k, j) to zero.
                    for(i = 0; i < n_comp; i++) {
                        // Note that the transpose of Wmat is accessed here (W' * W).
                        WTWH[k * cols + j] += WTW[k * n_comp + i] * Hmat_master[i * cols + j];
                    }
                }
            }

            // Update H.
            for(i = 0; i < n_comp * cols; i++) {
                Hmat_master[i] = Hmat_master[i] * (WTV[i] / (WTWH[i] + DBL_EPSILON));
            }

        }

        // TODO: Master send H to workers.
        // TODO: Workers receive H from master.

        // TODO:
        /* Update W. */
        /* -------------------------------------------------- */
        // Similar to updating H.

        // TODO: Master send W to workers.
        // TODO: Workers receive W from master.
    }


    // Free memory allocated for all matrices.

    if(rank == MASTER) {
        free(Wmat_master);
        free(Hmat_master);
        free(Wmat_worker);
        free(Hmat_worker);
        free(WTV);
        free(WTW);
        free(WTWH);
        free(VHT);
        free(HHT);
        free(WHHT);
        free(WTVblock_master);
        free(VHTblock_master);
    } else {
        free(Wmat_worker);
        free(Hmat_worker);
        free(Vcol);
        free(Vrow);
        free(WTVblock_worker);
        free(VHTblock_worker);
    }

    MPI_Finalize();

    return 0;
}

double randomDouble() {
    double r = (double) rand()/(double) RAND_MAX;
    return r;
}

