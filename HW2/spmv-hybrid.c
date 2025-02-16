#include <stdio.h>
#include "cmdline.h"
#include "input.h"
#include "config.h"
#include "timer.h"
#include "formats.h"

#include <omp.h>
#include <mpi.h>
                                                    
#define max(a,b) \
({ __typeof__ (a) _a = (a); \
   __typeof__ (b) _b = (b); \
 _a > _b ? _a : _b; })

#define min(a,b) \
({ __typeof__ (a) _a = (a); \
   __typeof__ (b) _b = (b); \
 _a < _b ? _a : _b; })     

void usage(int argc, char** argv)
{
    printf("Usage: %s [my_matrix.mtx]\n", argv[0]);
    printf("Note: my_matrix.mtx must be real-valued sparse matrix in the MatrixMarket file format.\n"); 
}

void accumulate_results(coo_matrix * coo, float* x, float* y, int process_number, int num_of_processes)
{
    int num_nonzeros = coo->num_nonzeros;
    int num_rows = coo->num_rows;

    // Step-1: Processes 1,3,5,7 send their y[] to processes 0,2,4,6 respectively, to be accumulated into their y[]
    // Step-2: Processes 2, 6 send their y[] to Processes 0, 4 respectively, to be accumulated into their y[]
    // Step-3: Process 4 sends its y[] to 0, to be accumulated into its y[]
    // Proces-0 will have the final y[]
    // ---------------------- Accumulate the results ---------------------------------

    // Step 1: Odd processes send to even processes
    if (process_number % 2 == 1 && process_number < num_of_processes) 
    {
        MPI_Send(y, num_rows, MPI_FLOAT, process_number - 1, 100, MPI_COMM_WORLD);
    }

    if (process_number % 2 == 0 && process_number + 1 < num_of_processes) 
    {
        float* temp = (float*)malloc(num_rows * sizeof(float));
        MPI_Recv(temp, num_rows, MPI_FLOAT, process_number + 1, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        #pragma omp parallel for
        for (int i = 0; i < num_rows; i++) {
            y[i] += temp[i];
        }
        free(temp);
    }
    MPI_Barrier(MPI_COMM_WORLD);    //*****

    // Step 2: Every fourth process (2, 6, 10, ...) sends to (0, 4, 8, ...)
    if (process_number % 4 == 2 && process_number < num_of_processes) 
    {
        MPI_Send(y, num_rows, MPI_FLOAT, process_number - 2, 100, MPI_COMM_WORLD);
    }

    if (process_number % 4 == 0 && process_number + 2 < num_of_processes) 
    {
        float* temp = (float*)malloc(num_rows * sizeof(float));
        MPI_Recv(temp, num_rows, MPI_FLOAT, process_number + 2, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        #pragma omp parallel for
        for (int i = 0; i < num_rows; i++) {
            y[i] += temp[i];
        }
        free(temp);
    }
    MPI_Barrier(MPI_COMM_WORLD);    //******

    // Step 3: Process 4, 8, 12, ... send to process 0
    if (process_number % 4 == 0 && process_number > 0 && process_number < num_of_processes) {
        MPI_Send(y, num_rows, MPI_FLOAT, 0, 100, MPI_COMM_WORLD);
    }

    if (process_number == 0) 
    {
        float* temp = (float*)malloc(num_rows * sizeof(float));
        // Receive from all processes that are multiples of 4 (except 0)
        for (int sender = 4; sender < num_of_processes; sender += 4) {
            MPI_Recv(temp, num_rows, MPI_FLOAT, sender, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            #pragma omp parallel for
            for (int i = 0; i < num_rows; i++) {
                y[i] += temp[i];
            }
        }
        free(temp);
    }
    MPI_Barrier(MPI_COMM_WORLD);    //******
        
    
    if (process_number != 0)    //------ Clear the y[] of all the Processes other than 0
    {
        memset(y, 0, num_rows * sizeof(float));
    }

}

// MIN_ITER, MAX_ITER, TIME_LIMIT, 
double benchmark_coo_spmv(coo_matrix * coo, float* x, float* y)
{
    //----------- MPI Process variables --------------
    int process_number;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_number);
    int num_of_processes;
    MPI_Comm_size(MPI_COMM_WORLD, &num_of_processes);


    int num_nonzeros = coo->num_nonzeros;
    int num_rows = coo->num_rows;

    // *** Timer Start *** : Only process-0 is responsible for measuring the time taken
    timer time_one_iteration;
    if (process_number == 0)
    {    
        timer_start(&time_one_iteration);
    }


    int subarray_length_per_process = num_nonzeros / num_of_processes;         // Number of processes = 8

    // A separate copy of y[] is created in the heap memory of each of the processes. 
    // The process with process_number iterates [ subarray_length_per_process * (process_number) : subarray_length_per_process * (process_number+1) ]
    #pragma omp parallel for
    for (int i = subarray_length_per_process * process_number; i < subarray_length_per_process * (process_number+1); i++)
    {   
        #pragma omp atomic
        y[coo->rows[i]] += coo->vals[i] * x[coo->cols[i]];
    }

    // Handle the remaining part of the array (The REMAINDER when num_nonzeros is divided by 8)
    // Done by final process: num_of_processes - 1
    if (process_number == num_of_processes-1)
    {
        if (num_nonzeros % num_of_processes != 0)   // Check if there is actually any remainder
        {
            #pragma omp parallel for
            for (int i = subarray_length_per_process * num_of_processes; i < num_nonzeros; i++)
            {
                #pragma omp atomic
                y[coo->rows[i]] += coo->vals[i] * x[coo->cols[i]];
            }
        }   
    }

    //------------------------------------------------ Y[] Accumulation Start --------------------------------
    accumulate_results(coo, x, y, process_number, num_of_processes);
    //------------------------------------------------ y[] Accumulation Done --------------------------------------------------------------------------------------------
    
    
    // Print estimated time
    double estimated_time;
    if (process_number == 0)
    {
        estimated_time = seconds_elapsed(&time_one_iteration);   // *** Timer End ***

        // Send the estimated time to all the other processes for consistency
        for (int p=1; p<num_of_processes; p++)
        {
            MPI_Send(&estimated_time, 1, MPI_DOUBLE, p, 100, MPI_COMM_WORLD); // Send y[] to process-0
        }

        //printf("estimated time for once %f\n", (float) estimated_time);     // Delete this
    }

    // Receive the estimated time from process-0 for consistency
    if (process_number != 0)
    {
        MPI_Recv(&estimated_time, 1, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);  // Receive y[] from process + 2
    }


    // -------------------------------------------- Average Calculation ----------------------------------------------
    // determine # of seconds dynamically
    int num_iterations ; //= MAX_ITER;
    // printf("\tPerforming %d iterations\n", num_iterations);
    // if (process_number == 0)
    // {
        num_iterations = 400;

        // if (estimated_time == 0)
        //     num_iterations = MAX_ITER;
        // else {
        //     num_iterations = min(MAX_ITER, max(MIN_ITER, (int) (TIME_LIMIT / estimated_time)) ); 
        // }
         printf("\tPerforming %d iterations\n", num_iterations);
    //}

    // time several SpMV iterations
    timer t;
    if (process_number == 0)
    {
        timer_start(&t);
    }

    
    for(int j = 0; j < num_iterations; j++)
    {
        // ******************************** Perform SpMV ************************************************

        // The process with process_number iterates [ subarray_length_per_process * process_number : subarray_length_per_process * (process_number+1) ]
        #pragma omp parallel for
        for (int i = subarray_length_per_process * process_number; i < subarray_length_per_process * (process_number+1); i++)
        {
            #pragma omp atomic   
            y[coo->rows[i]] += coo->vals[i] * x[coo->cols[i]];
        }

        // Handle the remaining part of the array (The REMAINDER when num_nonzeros is divided by 8)
        // Done by final process: num_of_processes - 1
        if (process_number == num_of_processes-1)
        {
            if (num_nonzeros % num_of_processes != 0)   // Check if there is actually any remainder
            {
                #pragma omp parallel for
                for (int i = subarray_length_per_process * num_of_processes; i < num_nonzeros; i++)
                {
                    #pragma omp atomic
                    y[coo->rows[i]] += coo->vals[i] * x[coo->cols[i]];
                }
            }   
        }


        //------------------------------------------------ y[] Accumulation Start --------------------------------------------------------------------------------------------
        accumulate_results(coo, x, y, process_number, num_of_processes);
        //------------------------------------------------ y[] Accumulation Done --------------------------------------------------------------------------------------------
    

        MPI_Barrier(MPI_COMM_WORLD);    // -- Barrier to synchronize Step-1 processes --
    }
    
    // ******************************** Completed SpMV ************************************************


    double msec_per_iteration;
    if (process_number == 0)
    {
        msec_per_iteration = milliseconds_elapsed(&t) / (double) num_iterations;
        double sec_per_iteration = msec_per_iteration / 1000.0;
        double GFLOPs = (sec_per_iteration == 0) ? 0 : (2.0 * (double) coo->num_nonzeros / sec_per_iteration) / 1e9;
        double GBYTEs = (sec_per_iteration == 0) ? 0 : ((double) bytes_per_coo_spmv(coo) / sec_per_iteration) / 1e9;
        printf("\tbenchmarking COO-SpMV: %8.4f ms ( %5.2f GFLOP/s %5.1f GB/s)\n", msec_per_iteration, GFLOPs, GBYTEs); 
    }

    return msec_per_iteration;
}

int main(int argc, char** argv)
{
    MPI_Init(NULL, NULL);   // ****************

    //----------- MPI Process variables --------------
    int process_number;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_number);
    int num_of_processes;
    MPI_Comm_size(MPI_COMM_WORLD, &num_of_processes);

    if (get_arg(argc, argv, "help") != NULL){
        usage(argc, argv);
        return 0;
    }

    char * mm_filename = NULL;
    if (argc == 1) {
        printf("Give a MatrixMarket file.\n");
        return -1;
    } else 
        mm_filename = argv[1];

    coo_matrix coo;
    read_coo_matrix(&coo, mm_filename);

    // fill matrix with random values: some matrices have extreme values, 
    // which makes correctness testing difficult, especially in single precision
    srand(13);
    for(int i = 0; i < coo.num_nonzeros; i++) {
        coo.vals[i] = 1.0 - 2.0 * (rand() / (RAND_MAX + 1.0)); 
        // coo.vals[i] = 1.0;
    }
    
    printf("\nfile=%s rows=%d cols=%d nonzeros=%d\n", mm_filename, coo.num_rows, coo.num_cols, coo.num_nonzeros);
    fflush(stdout);

#ifdef TESTING
//print in COO format
    printf("Writing matrix in COO format to test_COO ...");
    FILE *fp = fopen("test_COO", "w");
    fprintf(fp, "%d\t%d\t%d\n", coo.num_rows, coo.num_cols, coo.num_nonzeros);
    fprintf(fp, "coo.rows:\n");
    for (int i=0; i<coo.num_nonzeros; i++)
    {
      fprintf(fp, "%d  ", coo.rows[i]);
    }
    fprintf(fp, "\n\n");
    fprintf(fp, "coo.cols:\n");
    for (int i=0; i<coo.num_nonzeros; i++)
    {
      fprintf(fp, "%d  ", coo.cols[i]);
    }
    fprintf(fp, "\n\n");
    fprintf(fp, "coo.vals:\n");
    for (int i=0; i<coo.num_nonzeros; i++)
    {
      fprintf(fp, "%f  ", coo.vals[i]);
    }
    fprintf(fp, "\n");
    fclose(fp);
    printf("... done!\n");
#endif 

    //initialize host arrays
    float * x = (float*)malloc(coo.num_cols * sizeof(float));
    float * y = (float*)malloc(coo.num_rows * sizeof(float));

    for(int i = 0; i < coo.num_cols; i++) {
        x[i] = rand() / (RAND_MAX + 1.0); 
        // x[i] = 1;
    }
    for(int i = 0; i < coo.num_rows; i++)
        y[i] = 0;

    /* Benchmarking */
    double coo_gflops;
    coo_gflops = benchmark_coo_spmv(&coo, x, y);      // ------------********************----------------

    /* Test correctnesss */
// #ifdef TESTING
    if (process_number == 0)
    {
    FILE *fp = fopen("test_out", "w");
    printf("Writing x and y vectors ...");
    fp = fopen("test_x", "w");
    for (int i=0; i<coo.num_cols; i++)
    {
      fprintf(fp, "%f\n", x[i]);
    }
    fclose(fp);
    fp = fopen("test_y", "w");
    for (int i=0; i<coo.num_rows; i++)
    {
      fprintf(fp, "%f\n", y[i]);
    }
    fclose(fp);
    printf("... done!\n");
    }
// #endif

    delete_coo_matrix(&coo);
    free(x);
    free(y);

    MPI_Finalize();     // ****************

    return 0;
}

