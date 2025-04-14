#pragma once

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include "formats.h"
#include "mmio.h"
#include "../config.h"

static void swap_nnzs(coo_matrix *coo, int ind1, int ind2) 
{
    int eleind1;
    eleind1 = coo->rows[ind1];
    coo->rows[ind1] = coo->rows[ind2];
    coo->rows[ind2] = eleind1;
    eleind1 = coo->cols[ind1];
    coo->cols[ind1] = coo->cols[ind2];
    coo->cols[ind2] = eleind1;
    float val1 = coo->vals[ind1];
    coo->vals[ind1] = coo->vals[ind2];
    coo->vals[ind2] = val1;
}

static int cmp_indices(coo_matrix *coo, int loc1, int loc2)
{
    int eleind1 = coo->rows[loc1];
    int eleind2 = coo->rows[loc2];
    if(eleind1 < eleind2) {
        return -1;
    } else if(eleind1 > eleind2) {
        return 1;
    }
    return 0;
}


static void quicksort_coo(coo_matrix *coo, int l, int r)
{
    int i, j, p;
    if(r-l < 2) {
        return;
    }
    p = (l+r) / 2;
    for(i = l, j = r-1; ; ++i, --j) {
        while(cmp_indices(coo, i, p) < 0) {
            ++i;
        }
        while(cmp_indices(coo, p, j) < 0) {
            --j;
        }
        if(i >= j) {
            break;
        }
        swap_nnzs(coo, i, j);
        if(i == p) {
            p = j;
        } else if(j == p) {
            p = i;
        }
    }
    quicksort_coo(coo, l, i);
    quicksort_coo(coo, i, r);
}

void sort_coo(coo_matrix *coo) 
{
    quicksort_coo(coo, 0, coo->num_nonzeros);
}


void read_coo_matrix(coo_matrix *coo, const char * mm_filename)
{
    FILE * fid;
    MM_typecode matcode;
    
    fid = fopen(mm_filename, "r");

    if (fid == NULL){
        printf("Unable to open file %s\n", mm_filename);
        exit(1);
    }

    if (mm_read_banner(fid, &matcode) != 0){
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }

    if (!mm_is_valid(matcode)){
        printf("Invalid Matrix Market file.\n");
        exit(1);
    }

    if (!((mm_is_real(matcode) || mm_is_integer(matcode) || mm_is_pattern(matcode)) && mm_is_coordinate(matcode) && mm_is_sparse(matcode) ) ){
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        printf("Only sparse real-valued or pattern coordinate matrices are supported\n");
        exit(1);
    }

    int num_rows, num_cols, num_nonzeros;
    if ( mm_read_mtx_crd_size(fid,&num_rows,&num_cols,&num_nonzeros) !=0)
            exit(1);

    coo->num_rows     = (int) num_rows;
    coo->num_cols     = (int) num_cols;
    coo->num_nonzeros = (int) num_nonzeros;

    coo->rows = (int*)malloc(coo->num_nonzeros * sizeof(int));
    coo->cols = (int*)malloc(coo->num_nonzeros * sizeof(int));
    coo->vals = (float*)malloc(coo->num_nonzeros * sizeof(float));

    printf("Reading sparse matrix from file (%s):",mm_filename);
    fflush(stdout);

    if (mm_is_pattern(matcode)){
        // pattern matrix defines sparsity pattern, but not values
        for( int i = 0; i < coo->num_nonzeros; i++ ){
            assert(fscanf(fid, " %d %d \n", &(coo->rows[i]), &(coo->cols[i])) == 2);
            coo->rows[i]--;      //adjust from 1-based to 0-based indexing
            coo->cols[i]--;
            coo->vals[i] = 1.0;  //use value 1.0 for all nonzero entries 
        }
    } else if (mm_is_real(matcode) || mm_is_integer(matcode)){
        for( int i = 0; i < coo->num_nonzeros; i++ ){
            int I,J;
            double V;  // always read in a double and convert later if necessary
            
            assert(fscanf(fid, " %d %d %lf \n", &I, &J, &V) == 3);

            coo->rows[i] = (int) I - 1; 
            coo->cols[i] = (int) J - 1;
            coo->vals[i] = (float)  V;
        }
    } else {
        printf("Unrecognized data type\n");
        exit(1);
    }

    fclose(fid);
    printf(" done\n");

    if( mm_is_symmetric(matcode) ){ //duplicate off diagonal entries
        int off_diagonals = 0;
        for( int i = 0; i < coo->num_nonzeros; i++ ){
            if( coo->rows[i] != coo->cols[i] )
                off_diagonals++;
        }

        int true_nonzeros = 2*off_diagonals + (coo->num_nonzeros - off_diagonals);

        int* new_I = (int*)malloc(true_nonzeros * sizeof(int));
        int* new_J = (int*)malloc(true_nonzeros * sizeof(int));
        float * new_V = (float*)malloc(true_nonzeros * sizeof(float));

        int ptr = 0;
        for( int i = 0; i < coo->num_nonzeros; i++ ){
            if( coo->rows[i] != coo->cols[i] ){
                new_I[ptr] = coo->rows[i];  new_J[ptr] = coo->cols[i];  new_V[ptr] = coo->vals[i];
                ptr++;
                new_J[ptr] = coo->rows[i];  new_I[ptr] = coo->cols[i];  new_V[ptr] = coo->vals[i];
                ptr++;
            } else {
                new_I[ptr] = coo->rows[i];  new_J[ptr] = coo->cols[i];  new_V[ptr] = coo->vals[i];
                ptr++;
            }
        }       
         free(coo->rows); free(coo->cols); free(coo->vals);
         coo->rows = new_I;  coo->cols = new_J; coo->vals = new_V;      
         coo->num_nonzeros = true_nonzeros;
    } //end symmetric case

    // Sort the COO matrix
    sort_coo(coo);
}