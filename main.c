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
 * AUTHOR: Olavur Mortensen <olavurmortensen@gmail.com>, 2016.
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
        *displs,  // Used in MPI_Gatherv().
        *rcounts,  // Used in MPI_Gatherv().
        *displsW,
        *rcountsW,
        numworkers,
        errorcode,
        iter,
        comp_err,
        i, j, k, m;  // Index variables.

    double *Vcol,  // Subset of columns of V.
           *Vrow,  // Subset of rows of V.
           *Wmat,  // Dictionary, W.
           *Hmat,  // Activation, H.
           *WTV,
           *WTVblock,
           *WTVrecv,
           *WTW,
           *WTWH,
           *VHT,
           *VHTblock,
           *VHTrecv,
           *HHT,
           *WHHT,
           res2,  // Sum of squared residuals (each worker's contribution).
           res2_buff,
           rec,  // Reconstuction of a single point in the matrix V.
           err;  // Error, err = 0.5 * sqrt(sum(res2)).

    double randomDouble();  // Generates a random double between 0 and 1.

    MPI_Status status;

    // MPI initialization.
    errorcode = MPI_Init(&argc, &argv);
    if (errorcode != MPI_SUCCESS) {
        // NOTE: MPI_Abort is not the ideal way to quit MPI.
        printf("Error starting MPI program. Terminating.\n");
        MPI_Abort(MPI_COMM_WORLD, errorcode);
    }

    // Get process id (rank) and total number of processes (size).
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    numworkers = size - 1;

    if(size < 2) {
        // NOTE: MPI_Abort is not the ideal way to quit MPI.
        printf("dNMF needs at least 2 MPI processes, quitting.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(1);
    }

    rows = atoi(argv[1]);
    cols = atoi(argv[2]);
    n_comp = atoi(argv[3]);
    n_iter = atoi(argv[4]);
    comp_err = atoi(argv[5]);

    if(rows % numworkers != 0 || cols % numworkers != 0) {
        // NOTE: MPI_Abort is not the ideal way to quit MPI.
        if(rank == MASTER) {
            printf("Number of rows and columns in input matrix must be divisble by number of workers (number of processes minus 1). In other words, rows modulo numworkers must be 0, and cols modulo numworkers must be zero.\nQuitting...\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(1);
    }

    // TODO: account for when cols and/or rows does not split evenly among workers.
    // At the moment, assuming cols % numworkers = 0 and rows % numworkers = 0.
    bs_cols = (int) ((float) cols / numworkers);
    bs_rows = (int) ((float) rows / numworkers);

    // Used in MPI_Gatherv()
    // First for when updating H.
    displs = malloc(size * sizeof(int));
    rcounts = malloc(size * sizeof(int));
    displs[0] = 0;
    rcounts[0] = 0;  // Master prcess (0) sends no elements in operation => doesn't participate.
    for(i = 1; i < size; i++) {
        displs[i] = (i - 1) * n_comp * bs_cols;
        rcounts[i] = n_comp * bs_cols;
    }

    // Also for when updating W.
    displsW = malloc(size * sizeof(int));
    rcountsW = malloc(size * sizeof(int));
    displsW[0] = 0;
    rcountsW[0] = 0;  // Master prcess (0) sends no elements in operation => doesn't participate.
    for(i = 1; i < size; i++) {
        displsW[i] = (i - 1) * bs_rows * n_comp;
        rcountsW[i] = bs_rows * n_comp;
    }

    /* Allocate matrices. */
    /* ----------------------------------------------- */
    Wmat = malloc(rows * n_comp * sizeof(double));
    Hmat = malloc(n_comp * cols * sizeof(double));
    WTVblock = malloc(n_comp * bs_cols * sizeof(double));
    VHTblock = malloc(bs_rows * n_comp * sizeof(double));
    if(rank == MASTER) {
        WTV = malloc(n_comp * cols * sizeof(double));
        WTVrecv = malloc(n_comp * cols * sizeof(double));
        WTW = malloc(n_comp * n_comp * sizeof(double));
        WTWH = malloc(n_comp * cols * sizeof(double));
        VHT = malloc(rows * n_comp * sizeof(double));
        VHTrecv = malloc(rows * n_comp * sizeof(double));
        HHT = malloc(n_comp * n_comp * sizeof(double));
        WHHT = malloc(rows * n_comp * sizeof(double));

        // Randomly initialize Wmat and Hmat.
        srand(time(NULL));  // Seed for random number generator.
        
        for(i = 0; i < rows; i++) {
            for(k = 0; k < n_comp; k++) {
                Wmat[i * n_comp + k] = randomDouble() + DBL_EPSILON;
            }
        }

        for(i = 0; i < cols; i++) {
            for(k = 0; k < n_comp; k++) {
                Hmat[k * cols + i] = randomDouble() + DBL_EPSILON;
            }
        }
    }
    else {
        // Allocate matrices on workers.
        Vcol = malloc(rows * bs_cols * sizeof(double));
        Vrow = malloc(bs_rows * cols * sizeof(double));

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
    MPI_Bcast(Wmat, rows * n_comp, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(Hmat, n_comp * cols, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    /* Main algorithm loop. */
    for(iter = 0; iter < n_iter; iter++) {
        /* Compute the sum of squared residuals. */
        /* -------------------------------------------------- */
        if(iter % comp_err == 0) {
            if(rank != MASTER) {
                res2 = 0.0;
                for(i = 0; i < rows; i++) {
                    for(j = 0; j < bs_cols; j++) {
                        rec = 0.0;  // Initialize reconstruction of element (i, j) to zero.
                        for(k = 0; k < n_comp; k++) {
                            // Squared difference between V and its reconstruction at point (i, j).
                            rec += Wmat[i * n_comp + k] * Hmat[k * cols + (rank - 1) * bs_cols + j];
                        }
                        // Add the squared residual for the current position.
                        res2 += pow(Vcol[i * bs_cols + j] - rec, 2);
                    }
                }
            }

            // Sum all squared residuals on master.
            MPI_Reduce(&res2, &res2_buff, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);

            if(rank == MASTER) {
                // Compute and print reconstruction error.
                err = 0.5 * sqrt(res2_buff);
                printf("%.4e\n", err);
            }
        }

        /* Update H. */
        /* -------------------------------------------------- */
        // Compute WTVblock.
        if(rank != MASTER) {
            for(k = 0; k < n_comp; k++) {
                for(j = 0; j < bs_cols; j++) {
                    WTVblock[k * bs_cols + j] = 0.0;  // Initialize element (k, j) to zero.
                    for(i = 0; i < rows; i++) {
                        WTVblock[k * bs_cols + j] += Wmat[i * n_comp + k] * Vcol[i * bs_cols + j];
                    }
                }
            }
        }

        // Send blocks of WTV to master, concatenate into one array.
        MPI_Gatherv(WTVblock, n_comp * bs_cols, MPI_DOUBLE, WTVrecv, rcounts, displs, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);


        if(rank == MASTER) {

            // Re-arrange the values in the matrix WTVrecv into matrix WTV.
            // When WTVblock matrices are concatenated in MPI_Gatherv, the resulting matrix is not the right structure.
            // This re-arrangement is costly but necessary.
            for(i = 0; i < bs_cols; i++) {
                for(j = 0; j < numworkers; j++) {
                    for(k = 0; k < n_comp; k++) {
                        WTV[k * cols + j * bs_cols + i] = WTVrecv[j * n_comp * bs_cols + k * bs_cols + i];
                    }
                }
            }

            // Compute WTW = W' * W
            for(k = 0; k < n_comp; k++) {
                for(i = 0; i < n_comp; i++) {
                    WTW[k * n_comp + i] = 0.0;  // Initialize element (k, i) to zero.
                    for(j = 0; j < rows; j++) {
                        WTW[k * n_comp + i] += Wmat[j * n_comp + k] * Wmat[j * n_comp + i];
                    }
                }
            }

            // Compute WTWH = WTW * H
            for(k = 0; k < n_comp; k++) {
                for(j = 0; j < cols; j++) {
                    WTWH[k * cols + j] = 0.0;  // Initialize element (k, j) to zero.
                    for(i = 0; i < n_comp; i++) {
                        WTWH[k * cols + j] += WTW[k * n_comp + i] * Hmat[i * cols + j];
                    }
                }
            }

            // Update H.
            for(i = 0; i < n_comp * cols; i++) {
                Hmat[i] = Hmat[i] * (WTV[i] / (WTWH[i] + DBL_EPSILON));
            }

        }

        // Master send H to workers.
        MPI_Bcast(Hmat, n_comp * cols, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

        /* Update W. */
        /* -------------------------------------------------- */
        // Similar to updating H.
        // Compute VHTblock.
        if(rank != MASTER) {
            for(k = 0; k < n_comp; k++) {
                for(j = 0; j < bs_rows; j++) {
                    VHTblock[j * n_comp + k] = 0.0;  // Initialize element (k, j) to zero.
                    for(i = 0; i < cols; i++) {
                        // Note that the transpose of Hmat is accessed here.
                        VHTblock[j * n_comp + k] += Vrow[j * cols + i] * Hmat[k * cols + i];
                    }
                }
            }
        }

        // Send blocks of VHT to master, concatenate into one array.
        MPI_Gatherv(VHTblock, bs_rows * n_comp, MPI_DOUBLE, VHT, rcountsW, displsW, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

        if(rank == MASTER) {
            // Compute HHT = H * H'.
            for(k = 0; k < n_comp; k++) {
                for(i = 0; i < n_comp; i++) {
                    HHT[k * n_comp + i] = 0.0;  // Initialize element (k, i) to zero.
                    for(j = 0; j < cols; j++) {
                        // Note that the transpose of Hmat is accessed here (H * H').
                        HHT[k * n_comp + i] += Hmat[k * cols + j] * Hmat[i * cols + j];
                    }
                }
            }

            // Compute WHHT = W * HHT
            for(k = 0; k < n_comp; k++) {
                for(j = 0; j < rows; j++) {
                    WHHT[j * n_comp + k] = 0.0;  // Initialize element (k, j) to zero.
                    for(i = 0; i < n_comp; i++) {
                        WHHT[j * n_comp + k] += Wmat[j * n_comp + i] * HHT[k * n_comp + i];
                    }
                }
            }

            // Update W.
            for(i = 0; i < rows * n_comp; i++) {
                Wmat[i] = Wmat[i] * (VHT[i] / (WHHT[i] + DBL_EPSILON));
            }
        }

        // Master send W to workers.
        MPI_Bcast(Wmat, rows * n_comp, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    }

    // Compute reconstruction error one last time.
    if(rank != MASTER) {
        res2 = 0.0;
        for(i = 0; i < rows; i++) {
            for(j = 0; j < bs_cols; j++) {
                rec = 0.0;  // Initialize reconstruction of element (i, j) to zero.
                for(k = 0; k < n_comp; k++) {
                    // Squared difference between V and its reconstruction at point (i, j).
                    rec += Wmat[i * n_comp + k] * Hmat[k * cols + (rank - 1) * bs_cols + j];
                }
                // Add the squared residual for the current position.
                res2 += pow(Vcol[i * bs_cols + j] - rec, 2);
            }
        }
    }

    // Sum all squared residuals on master.
    MPI_Reduce(&res2, &res2_buff, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);

    if(rank == MASTER) {
        // Compute and print reconstruction error.
        err = 0.5 * sqrt(res2_buff);
        printf("Reconstruction error: %.4e\n", err);
    }

    // Free memory allocated for all matrices.
    free(Wmat);
    free(Hmat);
    free(displs);
    free(rcounts);
    free(displsW);
    free(rcountsW);
    free(WTVblock);
    free(VHTblock);
    if(rank == MASTER) {
        free(WTV);
        free(WTW);
        free(WTWH);
        free(VHT);
        free(HHT);
        free(WHHT);
    } else {
        free(Vcol);
        free(Vrow);
    }

    MPI_Finalize();

    return 0;
}

double randomDouble() {
    double r = (double) rand()/(double) RAND_MAX;
    return r;
}

