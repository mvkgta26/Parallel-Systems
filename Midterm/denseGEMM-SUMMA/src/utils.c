#include "utils.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/*
void matmul(float* A, float* B, float* C, int m, int n, int k) {
    // Initialize output matrix to zero
    //memset(C, 0, m * n * sizeof(float));
    
    // C[i,j] = sum(A[i,p] * B[p,j])
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                // A is m×k, B is k×n
                float a_val = A[i * k + p];       // A[i,p]
                float b_val = B[p * n + j];       // B[p,j]
                sum += a_val * b_val;
            }
            C[i * n + j] += sum; // I made changes here to make addition accumulate into C[]
        }
    }
}
*/

void matmul(float *A, float *B, float *C, int m, int n, int k) {
    // C[i,j] = sum(A[i,p] * B[p,j])
    for (int i = 0; i < m; i++) {
      for (int p = 0; p < k; p++) {
        for (int j = 0; j < n; j++) {
          C[i * n + j] += A[i * k + p] * B[p * n + j];
        }
      }
    }
  }

void verify_result(float* C_global, float* A, float* B, int m, int n, int k) {
    int errors = 0;
    float tolerance = 1e-5;
    // Perform reference matrix multiplication
    float* C_ref = (float*)calloc(m * n, sizeof(float));
    if (!C_ref) {
        printf("Error: Failed to allocate memory for C_ref\n");
        return;
    }

    // Compute reference result
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += A[i*k + p] * B[p*n + j];
            }
            C_ref[i*n + j] = sum;
        }
    }

    // Compute detailed error statistics
    float max_error = 0.0f;
    float avg_error = 0.0f;
    int max_error_index = -1;
    int max_error_i = -1, max_error_j = -1;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int idx = i*n + j;
            float curr_error = fabs(C_global[idx] - C_ref[idx]);
            avg_error += curr_error;
            
            if (curr_error > max_error) {
                max_error = curr_error;
                max_error_index = idx;
                max_error_i = i;
                max_error_j = j;
            }
            
            if (curr_error > tolerance) {
                errors++;                if (errors <= 5) {
                    printf("Error at position [%d,%d]: C_global=%.6f, C_ref=%.6f, diff=%.6f\n",
                           i, j, C_global[idx], C_ref[idx], curr_error);
                }
            }
        }
    }
    avg_error /= (m * n);

    // Print detailed verification results
    printf("\nVerification Results:\n");
    printf("  Total Elements: %d\n", m * n);
    printf("  Errors: %d (%.2f%%)\n", errors, 100.0f * errors / (m * n));
    printf("  Max Error: %e at position [%d,%d] (index %d)\n", 
           max_error, max_error_i, max_error_j, max_error_index);
    printf("  Average Error: %e\n", avg_error);

    free(C_ref);
}

// Utility function to load matrix from file or generate
float* generate_matrix_A(int rows, int cols, int rank) {
    float* matrix = malloc(rows * cols * sizeof(float));
    srand(42);
    
    for (int i = 0; i < rows * cols; i++) {
        //matrix[i] = 3.5;
        matrix[i] = (float)rand() / RAND_MAX;
    }
    
    return matrix;
}


float* generate_matrix_B(int rows, int cols, int rank) {
    float* matrix = malloc(rows * cols * sizeof(float));
    srand(142);

    for (int i = 0; i < rows * cols; i++) {
        // matrix[i] = 2.5;//(float)rand() / RAND_MAX;
        matrix[i] = (float)rand() / RAND_MAX;
    }

    return matrix;
}


// -- Added By me -- Print a normal matrix
void print_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%8.2f ", mat[i * cols + j]);
        }
        printf("\n");
    }
}