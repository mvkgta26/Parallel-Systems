// utils.h
#ifndef UTILS_H
#define UTILS_H

void matmul(float* A, float* B, float* C, int m, int n, int k);
void verify_result(float* C_global, float* A, float* B, int m, int n, int k);
float* generate_matrix_A(int rows, int cols, int rank);
float* generate_matrix_B(int rows, int cols, int rank);

#endif