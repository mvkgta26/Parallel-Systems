
#pragma once

// COOrdinate matrix (aka IJV or Triplet format)
typedef struct coo_matrix
{
    int num_rows, num_cols, num_nonzeros;
    int * rows;  //row indices
    int * cols;  //column indices
    float * vals;  //nonzero values
} coo_matrix;


void delete_coo_matrix(coo_matrix* coo){
    free(coo->rows);   free(coo->cols);   free(coo->vals);
}

size_t bytes_per_coo_spmv(const coo_matrix * coo)
{
    size_t bytes = 0;
    bytes += 2*sizeof(int) * coo->num_nonzeros; // row and column indices
    bytes += 2*sizeof(float) * coo->num_nonzeros; // A[i,j] and x[j]

    size_t * occupied_rows = (size_t*)malloc(coo->num_rows * sizeof(size_t));
    for(size_t n = 0; n < coo->num_rows; n++)
    	occupied_rows[n] = 0;

    for(size_t n = 0; n < coo->num_nonzeros; n++)
        occupied_rows[coo->rows[n]] = 1;
    for(size_t n = 0; n < coo->num_rows; n++)
        if(occupied_rows[n] == 1)
            bytes += 2*sizeof(float);            // y[i] = y[i] + ...
    return bytes;
}

