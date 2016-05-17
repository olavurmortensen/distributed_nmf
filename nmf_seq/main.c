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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>

int main(int argc, char* argv[]) {
    int rows,  // Rows in V matrix.
        cols, // Columns in V matrix.
        n_comp,  // Number of NMF components.
        n_iter,  // Number of iterations of the NMF algorithm.
        comp_err,  // Compute reconstruction error every comp_err iterations.
        print_err,  // Whether or not to print the reconstruction error to console.
        memory_footprint,
        iter,
        i, j, k, m;  // Index variables.

    double *V,
           *Wmat,  // Dictionary, W.
           *Hmat,  // Activation, H.
           *WTV,
           *WTW,
           *WTWH,
           *VHT,
           *HHT,
           *WHHT,
           res2,  // Sum of squared residuals (each worker's contribution).
           rec,  // Reconstuction of a single point in the matrix V.
           err;  // Error, err = 0.5 * sqrt(sum(res2)).

    clock_t start, stop;

    double randomDouble();  // Generates a random double between 0 and 1.

    rows = atoi(argv[1]);
    cols = atoi(argv[2]);
    n_comp = atoi(argv[3]);
    n_iter = atoi(argv[4]);
    comp_err = atoi(argv[5]);
    print_err = atoi(argv[6]);

    start = clock();

    /* Allocate matrices. */
    /* ----------------------------------------------- */
    memory_footprint = 4 * (rows * n_comp);  // TODO: Calculate the memory usage of the program:
    V = malloc(rows * cols * sizeof(double));
    Wmat = malloc(rows * n_comp * sizeof(double));
    Hmat = malloc(n_comp * cols * sizeof(double));
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
            Wmat[i * n_comp + k] = randomDouble() + DBL_EPSILON;
        }
    }

    for(i = 0; i < cols; i++) {
        for(k = 0; k < n_comp; k++) {
            Hmat[k * cols + i] = randomDouble() + DBL_EPSILON;
        }
    }

    // Initialize V.
    // In reality, Vcol and Vrow would be read from hard drives on the cluster, or something along those lines.
    for(i = 0; i < rows; i++) {
        for(j = 0; j < cols; j++) {
            V[i * cols + j] = 1.0;
        }
    }

    /* Main algorithm loop. */
    for(iter = 0; iter < n_iter; iter++) {
        /* Compute the sum of squared residuals. */
        /* -------------------------------------------------- */
        if(iter % comp_err == 0) {
            res2 = 0.0;
            for(i = 0; i < rows; i++) {
                for(j = 0; j < cols; j++) {
                    rec = 0.0;  // Initialize reconstruction of element (i, j) to zero.
                    for(k = 0; k < n_comp; k++) {
                        // Squared difference between V and its reconstruction at point (i, j).
                        rec += Wmat[i * n_comp + k] * Hmat[k * cols + j];
                    }
                    // Add the squared residual for the current position.
                    res2 += pow(V[i * cols + j] - rec, 2);
                }
            }

            // Compute and print reconstruction error.
            err = 0.5 * sqrt(res2);
            if(print_err) {
                printf("%.4e\n", err);
            }
        }

        /* Update H. */
        /* -------------------------------------------------- */
        // Compute WTV.
        for(k = 0; k < n_comp; k++) {
            for(j = 0; j < cols; j++) {
                WTV[k * cols + j] = 0.0;  // Initialize element (k, j) to zero.
                for(i = 0; i < rows; i++) {
                    WTV[k * cols + j] += Wmat[i * n_comp + k] * V[i * cols + j];
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

        /* Update W. */
        /* -------------------------------------------------- */
        // Similar to updating H.
        // Compute VHT.
        for(k = 0; k < n_comp; k++) {
            for(j = 0; j < rows; j++) {
                VHT[j * n_comp + k] = 0.0;  // Initialize element (k, j) to zero.
                for(i = 0; i < cols; i++) {
                    // Note that the transpose of Hmat is accessed here.
                    VHT[j * n_comp + k] += V[j * cols + i] * Hmat[k * cols + i];
                }
            }
        }

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

    // Compute reconstruction error one last time.
    res2 = 0.0;
    for(i = 0; i < rows; i++) {
        for(j = 0; j < cols; j++) {
            rec = 0.0;  // Initialize reconstruction of element (i, j) to zero.
            for(k = 0; k < n_comp; k++) {
                // Squared difference between V and its reconstruction at point (i, j).
                rec += Wmat[i * n_comp + k] * Hmat[k * cols + j];
            }
            // Add the squared residual for the current position.
            res2 += pow(V[i * cols + j] - rec, 2);
        }
    }

    // Compute and print reconstruction error.
    err = 0.5 * sqrt(res2);
    if(print_err) {
        printf("%.4e\n", err);
    }

    // Free memory allocated for all matrices.
    free(V);
    free(Wmat);
    free(Hmat);
    free(WTV);
    free(WTW);
    free(WTWH);
    free(VHT);
    free(HHT);
    free(WHHT);

    stop = clock();

    printf("%f\n", (double)(stop - start) / CLOCKS_PER_SEC);

    return 0;
}

double randomDouble() {
    double r = (double) rand()/(double) RAND_MAX;
    return r;
}

