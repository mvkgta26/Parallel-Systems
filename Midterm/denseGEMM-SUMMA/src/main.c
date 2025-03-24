#include "summa_opts.h"
#include "utils.h"
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <timer.h>

void distribute_matrix_blocks(int rank, float* A, float* B, float* A_local, float* B_local, int p, int m, int n, int k, int block_m, int block_n, int block_k, float* B_temp) 
{
  // TODO: Implement matrix block distribution logic
  //Define the function signature and everything else

  // **---- Distribute B -----
  
  // -- Split and send from root --
  if (rank == 0)
  {
    printf("k, block_k : %d %d \n", k, block_k);
    for (int row = 0; row < k; row++)
    {
      // In each row, send 'p' blocks of block_n sized elements one by one, to the appropriate receving process
      for (int i = 0; i < p; i++)
      {
        int y = row / block_k;   // row/block_k gives the y-coordinate of the process mapped to the B-block that this row belongs to
        int x = i;  // Since i is used to iterate block_n steps, it is basically same as the x-coordinate of the proess that handles the B[] sub-block to which the current block belongs
        

        // If the receiving process is root itself
        if ( y*p + x == 0)
        {
          memcpy(B_local + row * block_n, B + row * n + i * block_n, block_n * sizeof(float));
        }
        
        //--memcpy(B_temp, B_local, block_k * block_n * sizeof(float));
        
        // B[row] + i*block_n is the starting address of the i'th block in the row
        // y*p + x is the rank of the process with coordinates x,y. We are sending this row-block to the process with coordinates (x,y)
        // B + row * n + i * block_n => Pointer to first element at row=row and ith block_n block 
        MPI_Send( B + row * n + i * block_n, block_n, MPI_FLOAT, y*p + x, 0, MPI_COMM_WORLD );
        // B[row] + i*block_n
        //B + row * n + i * block_n
      }
    }
  }

  // -- All the other processes receive B-local --
  else
  {
    // Iterate the rows in B_local[]
    for (int row = 0; row < block_k; row++)
    {
      // B[row] + i*block_n is the starting address of the i'th block in the row
      // y*p + x is the rank of the process with coordinates x,y. We are sending this row-block to the process with coordinates (x,y)
      //MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)
      MPI_Recv( B_local + row * block_n, block_n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

  }


  // **---- Distribute A -----
  
  // -- Split and send from root --
  if (rank == 0)
  {
    for (int row = 0; row < m; row++)
    {
      // In each row, send 'p' blocks of block_n sized elements one by one, to the appropriate receving process
      for (int i = 0; i < p; i++)
      {
        int y = row / block_m;   // row/block_k gives the y-coordinate of the process mapped to the B-block that this row belongs to
        int x = i;  // Since i is used to iterate block_n steps, it is basically same as the x-coordinate of the proess that handles the B[] sub-block to which the current block belongs
        

        // If the receiving process is root itself
        if ( y*p + x == 0)
        {
          memcpy(A_local + row * block_k, A + row * k + i * block_k, block_k * sizeof(float));
        }
        
        // A[row] + i*block_k is the starting address of the i'th block in the row
        // y*p + x is the rank of the process with coordinates x,y. We are sending this row-block to the process with coordinates (x,y)
        // A + row * k + i * block_k => Pointer to first element at row=row and ith block_k block 
        MPI_Send( A + row * k + i * block_k, block_k, MPI_FLOAT, y*p + x, 0, MPI_COMM_WORLD );
        // A[row] + i*block_k
        //A + row * k + i * block_k
      }
    }
  }

  // -- All the other processes receive A-local --
  else
  {
    // Iterate the rows in A_local[]
    for (int row = 0; row < block_m; row++)
    {
      // A[row] + i*block_k is the starting address of the i'th block in the row
      // y*p + x is the rank of the process with coordinates x,y. We are sending this row-block to the process with coordinates (x,y)
      //MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)
      MPI_Recv( A_local + row * block_k, block_k, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
  
}



void distribute_matrix_blocks_stationary_a(int rank, float* A, float* B, float* A_local_block, float* A_local_slab, float* B_local, int p, int m, int n, int k, int block_m, int block_n, int block_k, float* B_temp, MPI_Comm row_comm) 
{
  // TODO: Implement matrix block distribution logic
  //Define the function signature and everything else

  // **---- Distribute B -----
  
  // -- Split and send from root --
  if (rank == 0)
  {
    printf("k, block_k : %d %d \n", k, block_k);
    for (int row = 0; row < k; row++)
    {
      // In each row, send 'p' blocks of block_n sized elements one by one, to the appropriate receving process
      for (int i = 0; i < p; i++)
      {
        int y = row / block_k;   // row/block_k gives the y-coordinate of the process mapped to the B-block that this row belongs to
        int x = i;  // Since i is used to iterate block_n steps, it is basically same as the x-coordinate of the proess that handles the B[] sub-block to which the current block belongs
        
        // If the receiving process is root itself
        if ( y*p + x == 0)
        {
          memcpy(B_local + row * block_n, B + row * n + i * block_n, block_n * sizeof(float));
        }
        
        
        // B[row] + i*block_n is the starting address of the i'th block in the row
        // y*p + x is the rank of the process with coordinates x,y. We are sending this row-block to the process with coordinates (x,y)
        // B + row * n + i * block_n => Pointer to first element at row=row and ith block_n block 
        MPI_Send( B + row * n + i * block_n, block_n, MPI_FLOAT, y*p + x, 0, MPI_COMM_WORLD );
        // B[row] + i*block_n
        //B + row * n + i * block_n
      }
    }
  }

  // -- All the other processes receive B-local --
  else
  {
    // Iterate the rows in B_local[]
    for (int row = 0; row < block_k; row++)
    {
      // B[row] + i*block_n is the starting address of the i'th block in the row
      // y*p + x is the rank of the process with coordinates x,y. We are sending this row-block to the process with coordinates (x,y)
      //MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)
      MPI_Recv( B_local + row * block_n, block_n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

  }


  // **---- Distribute A -----
  
  // -- Split and send from root --
  if (rank == 0)
  {
    for (int row = 0; row < m; row++)
    {
      // In each row, send 'p' blocks of block_n sized elements one by one, to the appropriate receving process
      for (int i = 0; i < p; i++)
      {
        int y = row / block_m;   // row/block_k gives the y-coordinate of the process mapped to the B-block that this row belongs to
        int x = i;  // Since i is used to iterate block_n steps, it is basically same as the x-coordinate of the proess that handles the B[] sub-block to which the current block belongs
        

        // If the receiving process is root itself
        if ( y*p + x == 0)
        {
          memcpy(A_local_block + row * block_k, A + row * k + i * block_k, block_k * sizeof(float));
        }
        
        // A[row] + i*block_k is the starting address of the i'th block in the row
        // y*p + x is the rank of the process with coordinates x,y. We are sending this row-block to the process with coordinates (x,y)
        // A + row * k + i * block_k => Pointer to first element at row=row and ith block_k block 
        MPI_Send( A + row * k + i * block_k, block_k, MPI_FLOAT, y*p + x, 0, MPI_COMM_WORLD );
        // A[row] + i*block_k
        //A + row * k + i * block_k
      }
    }
  }

  // -- All the other processes receive A-local --
  else
  {
    // Iterate the rows in A_local[]
    for (int row = 0; row < block_m; row++)
    {
      // A[row] + i*block_k is the starting address of the i'th block in the row
      // y*p + x is the rank of the process with coordinates x,y. We are sending this row-block to the process with coordinates (x,y)
      //MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)
      MPI_Recv( A_local_block + row * block_k, block_k, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  // Gather all the A_local_blocks into a 2D array in row-rank-order
  // A_local_slab[0] ==> A_local_block 0 of process 0
  int chunk_size = block_m * block_k;
  MPI_Allgather(A_local_block, chunk_size, MPI_FLOAT, A_local_slab, chunk_size, MPI_FLOAT, row_comm);
  
}



void gather_matrix_blocks(float *local_matrix, float *matrix, int full_rows, int full_cols,
  int block_rows, int block_cols, int num_process, MPI_Comm grid_comm)
{
  MPI_Datatype block_type;

  // 1. Create the subarray type for structured gathering
  {
    MPI_Datatype temp_type;
    int sizes[2] = {full_rows, full_cols};      // Full matrix dimensions
    int subsizes[2] = {block_rows, block_cols}; // Block dimensions
    int starts[2] = {0, 0};                     // Always starts from (0,0), displacement handles offsets

    MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &temp_type);
    MPI_Type_create_resized(temp_type, 0, sizeof(float), &block_type);
    MPI_Type_commit(&block_type);
    MPI_Type_free(&temp_type);
  }

  // 2. Prepare recvcounts and displacements (only used on rank 0)
  int *recvcounts = NULL;
  int *displacements = NULL;

  if (matrix != NULL) // Only allocate on root (rank 0)
  {
    recvcounts = (int *)malloc(num_process * sizeof(int));
    displacements = (int *)malloc(num_process * sizeof(int));

    for (int i = 0; i < num_process; i++)
    {
    int proc_coords[2];
    MPI_Cart_coords(grid_comm, i, 2, proc_coords);

    recvcounts[i] = 1; // Each process sends one block

    // Compute displacement for 2D row-major storage
    int row_idx = proc_coords[0] * block_rows;
    int col_idx = proc_coords[1] * block_cols;
    displacements[i] = row_idx * full_cols + col_idx;
    }
  }

  // 3. Perform the gather operation
  MPI_Gatherv(local_matrix, block_rows * block_cols, MPI_FLOAT, // Send raw block
  matrix, recvcounts, displacements, block_type,    // Receive structured blocks
  0, grid_comm);

  // Clean
  if (matrix != NULL)
  {
    free(recvcounts);
    free(displacements);
  }

  MPI_Type_free(&block_type);
}



void summa_stationary_a(int m, int n, int k, int total_process_count, int rank) 
{

  //--Grid setup
  // TODO: Initialize grid variables
  

  // MPI_COMM_WORLD : Using this communicator, each process can communicate with ALL the processes creates
  MPI_Comm grid_comm;     // Grid communicator: Using this communicator, each process can only communicate with their neighbors (Periodicity=1 would also allow wraparound of neighbors). (Although bast will send to all processes in the grid)
  MPI_Comm col_comm;    // Column communicator: Using this communicator, each process can communicate with all the processes in the column in the grid 
  MPI_Comm row_comm;    // Row Communicator:  Using this communicator, each process can communicate with all the processes in the row in the grid 



  //--Create 2D process grid
  // TODO: Create a 2D Cartesian communicator

  int num_dimensions = 2;  // 2D process grid
  int p = sqrt(total_process_count); // Process grid side length. Totally p*p processes
  int dims[2] = {p,p};    // 2 Dimensions of p * p
  int periods[2] = {0, 0};    // Periodicity : periods[0] = Periodicity in x-dimension. periods[1] = Periodicity in y-dimension. 0 = No periodicity = No Wrap Around
  // 0 : Flag: Dont rearrange ranks within the communicator for optimisation
  MPI_Cart_create(MPI_COMM_WORLD, num_dimensions, dims, periods, 0, &grid_comm);


  //--Get process coordinates
  // TODO: Get the coordinates of the process in the grid

  // Obtain the (x,y) coordinates of the current process in the process grid, store it in the coords[] array
  int coords[2];
  MPI_Cart_coords(grid_comm, rank, num_dimensions, coords);


  //--Create row and column communicators
  // TODO: Create row and column communicators for the grid

  // The below function creates a new communicator object and assigns it to col_comm variable
  // 2nd paramter is colour: All the processes of same colour have the same communicator
  // Here: Assign color = rank % p = the column to which the process belongs to in the process grid.
  // All the processes belonging to the same column share the same communicator 
  int color = rank % p;
  MPI_Comm_split(grid_comm, color, rank, &col_comm);


  // The below function creates a new communicator object and assigns it to row_comm variable
  // 2nd paramter is colour: All the processes of same colour have the same communicator
  // Here: Assign color = rank / p = the row to which the process belongs to in the process grid.
  // All the processes belonging to the same column share the same communicator 
  color = rank / p;
  MPI_Comm_split(grid_comm, color, rank, &row_comm);


  // Determine local block sizes
  // TODO: Calculate the local block sizes for each process

  // Divided Block Dimensions
  int block_m = ceil(m/p);   //ceil(M/p)     // ---- check is variables are available ????
  int block_n = ceil(n/p);  //ceil(N/p)
  int block_k = ceil(k/p);  //ceil(K/p)



  // Generate random matrices on root process
  // TODO: Generate random matrices A and B on the root process
  float* A, *B;
  if (rank == 0)
  {
    A = generate_matrix_A(m, k, rank);  // rank is unused
    B = generate_matrix_B(k, n, rank);
  }

  // Allocate local matrices
  // TODO: Allocate memory for local matrices
  int B_size_per_process = block_k * block_n; // num_rows * num_cols
  int A_block_size_per_process = block_m * block_k;
  int A_slab_size_per_process = block_m * k;
  int C_size_per_process = block_m * block_n;
  float* B_local = (float*) calloc(B_size_per_process, sizeof(float));
  float* B_temp = (float*) calloc(B_size_per_process, sizeof(float));
  float* A_local_block = (float*) calloc(A_block_size_per_process, sizeof(float));
  float* A_local_slab = (float*) calloc(A_slab_size_per_process, sizeof(float));


  float* C_local = (float*) calloc(C_size_per_process, sizeof(float));
  float* C = (float*) calloc(m*n, sizeof(float));

  // Distribute matrix blocks
  // TODO: Distribute matrix blocks to all processes


  timer distribute_time;
  if (rank == 0)
  {    
      timer_start(&distribute_time);
  }

  // Distribute matrix blocks with proper arguments for all processes
  // TODO: Call distribute_matrix_blocks with appropriate arguments
  distribute_matrix_blocks_stationary_a(rank, A, B, A_local_block, A_local_slab, B_local, p, m, n, k, block_m, block_n, block_k, B_temp, row_comm);


  double distribute_duration;
  if (rank == 0)
  {
    distribute_duration = seconds_elapsed(& distribute_time);   // *** Timer End ***

    printf("Distribute run time: %f\n", (float) distribute_duration);     // Delete this
  }
 

  // SUMMA computation
  // TODO: Implement the SUMMA computation
  //You can use the function in utils.c to perform the matrix multiplication

  for (int j=0; j<p; j++)
  { 

    int col_rank;
    MPI_Comm_rank(col_comm, &col_rank);

    //int root_rank;

    // Iterate the rows of the process grid
    // All processes in row-j, send to all processes in their respective columns
    int root = -1;
    if (coords[0] == j)
    {       
      // Copy b_local into b_temp of the process j
      memcpy( B_temp, B_local, block_k * block_n * sizeof(float) );
    }  

    // Broadcast B_temp of the current process into the B_temp of all the processes in the column of process-j
    MPI_Bcast( B_temp, block_k * block_n, MPI_FLOAT, j, col_comm );
    
    matmul(A_local_slab + j * block_m * block_k, B_temp, C_local, block_m, block_n, block_k);

  }


  MPI_Barrier(MPI_COMM_WORLD);


  timer gather_time;
  if (rank == 0)
  {    
      timer_start(&gather_time);
  }
  // Gather results
  // TODO: Gather the results from all processes
  gather_matrix_blocks(C_local, C, m, n, block_m, block_n, total_process_count, grid_comm);

  double gather_duration;
  if (rank == 0)
  {
    gather_duration = seconds_elapsed(&gather_time);   // *** Timer End ***

    printf("Gather run time: %f\n", (float) gather_duration);     // Delete this

    printf("Total communication time: %f \n", (float) gather_duration + (float) distribute_duration);

  }


  // Verify results
  // TODO: Verify the correctness of the results
  
    if (rank == 0)
    {
      verify_result(C, A, B, m, n, k);
    }
}

void summa_stationary_c(int m, int n, int k, int total_process_count, int rank) 
{

  //--Grid setup
  // TODO: Initialize grid variables
  

  // MPI_COMM_WORLD : Using this communicator, each process can communicate with ALL the processes creates
  MPI_Comm grid_comm;     // Grid communicator: Using this communicator, each process can only communicate with their neighbors (Periodicity=1 would also allow wraparound of neighbors). (Although bast will send to all processes in the grid)
  MPI_Comm col_comm;    // Column communicator: Using this communicator, each process can communicate with all the processes in the column in the grid 
  MPI_Comm row_comm;    // Row Communicator:  Using this communicator, each process can communicate with all the processes in the row in the grid 



  //--Create 2D process grid
  // TODO: Create a 2D Cartesian communicator

  int num_dimensions = 2;  // 2D process grid
  int p = sqrt(total_process_count); // Process grid side length. Totally p*p processes
  int dims[2] = {p,p};    // 2 Dimensions of p * p
  int periods[2] = {0, 0};    // Periodicity : periods[0] = Periodicity in x-dimension. periods[1] = Periodicity in y-dimension. 0 = No periodicity = No Wrap Around
  // 0 : Flag: Dont rearrange ranks within the communicator for optimisation
  MPI_Cart_create(MPI_COMM_WORLD, num_dimensions, dims, periods, 0, &grid_comm);


  //--Get process coordinates
  // TODO: Get the coordinates of the process in the grid

  // Obtain the (x,y) coordinates of the current process in the process grid, store it in the coords[] array
  int coords[2];
  MPI_Cart_coords(grid_comm, rank, num_dimensions, coords);


  //--Create row and column communicators
  // TODO: Create row and column communicators for the grid

  // The below function creates a new communicator object and assigns it to col_comm variable
  // 2nd paramter is colour: All the processes of same colour have the same communicator
  // Here: Assign color = rank % p = the column to which the process belongs to in the process grid.
  // All the processes belonging to the same column share the same communicator 
  int color = rank % p;
  MPI_Comm_split(grid_comm, color, rank, &col_comm);


  // The below function creates a new communicator object and assigns it to row_comm variable
  // 2nd paramter is colour: All the processes of same colour have the same communicator
  // Here: Assign color = rank / p = the row to which the process belongs to in the process grid.
  // All the processes belonging to the same column share the same communicator 
  color = rank / p;
  MPI_Comm_split(grid_comm, color, rank, &row_comm);


  // Determine local block sizes
  // TODO: Calculate the local block sizes for each process

  // Divided Block Dimensions
  int block_m = ceil(m/p);   //ceil(M/p)     // ---- check is variables are available ????
  int block_n = ceil(n/p);  //ceil(N/p)
  int block_k = ceil(k/p);  //ceil(K/p)



  // Generate random matrices on root process
  // TODO: Generate random matrices A and B on the root process
  float* A, *B;
  if (rank == 0)
  {
    A = generate_matrix_A(m, k, rank);  // rank is unused
    B = generate_matrix_B(k, n, rank);
  }

  // Allocate local matrices
  // TODO: Allocate memory for local matrices
  int B_size_per_process = block_k * block_n; // num_rows * num_cols
  int A_size_per_process = block_m * block_k;
  int C_size_per_process = block_m * block_n;
  float* B_local = (float*) calloc(B_size_per_process, sizeof(float));
  float* B_temp = (float*) calloc(B_size_per_process, sizeof(float));
  float* A_local = (float*) calloc(A_size_per_process, sizeof(float));
  float* A_temp = (float*) calloc(A_size_per_process, sizeof(float));

  float* C_local = (float*) calloc(C_size_per_process, sizeof(float));


  timer distribute_time;
  if (rank == 0)
  {    
      timer_start(&distribute_time);
  }

  // Distribute matrix blocks
  // TODO: Distribute matrix blocks to all processes

  // Distribute matrix blocks with proper arguments for all processes
  // TODO: Call distribute_matrix_blocks with appropriate arguments
  distribute_matrix_blocks(rank, A, B, A_local, B_local, p, m, n, k, block_m, block_n, block_k, B_temp);

  double distribute_duration;
  if (rank == 0)
  {
    distribute_duration = seconds_elapsed(& distribute_time);   // *** Timer End ***

    printf("Distribute run time: %f\n", (float) distribute_duration);     // Delete this
  }


  // SUMMA computation
  // TODO: Implement the SUMMA computation
  //You can use the function in utils.c to perform the matrix multiplication

  for (int k=0; k<p; k++)
  { 

    if (coords[0] == k)
    { 
      // Copy b_local into b_temp of the process k
      memcpy( B_temp, B_local, block_k * block_n * sizeof(float) );
    }

    // Iterate the rows of the process grid
    // All processes in row-k, send to all processes in their respective columns
    if (coords[1] == k)
    { 
      // Copy b_local into b_temp of the process k
      memcpy( A_temp, A_local, block_m * block_k * sizeof(float) );
    }

    // Broadcast A_temp of the current process into the B_temp of all the processes in the column of process-k
    MPI_Bcast( A_temp, block_m * block_k, MPI_FLOAT, k, row_comm );

    // Broadcast B_temp of the current process into the B_temp of all the processes in the column of process-k
    MPI_Bcast( B_temp, block_k * block_n, MPI_FLOAT, k, col_comm );


    
    matmul(A_temp, B_temp, C_local, block_m, block_n, block_k);

  }


  MPI_Barrier(MPI_COMM_WORLD);


  // Gather results
  // TODO: Gather the results from all processes

  MPI_Barrier(MPI_COMM_WORLD); // -- Barrier before gathering all the results

  // Create C_root to gather all the results
  float* C;
  if (rank == 0)
  {
    C = (float*) calloc(m * n, sizeof(float));
  }
  

  timer gather_time;
  if (rank == 0)
  {    
      timer_start(&gather_time);
  }

  // Gather results
  // TODO: Gather the results from all processes
  gather_matrix_blocks(C_local, C, m, n, block_m, block_n, total_process_count, grid_comm);


  double gather_duration;
  if (rank == 0)
  {
    gather_duration = seconds_elapsed(&gather_time);   // *** Timer End ***

    printf("Gather run time: %f\n", (float) gather_duration);     
    printf("Total communication time: %f \n", (float) gather_duration + (float) distribute_duration);
  }

  // Verify results
  // TODO: Verify the correctness of the results
  if (rank == 0)
  {
    verify_result(C, A, B, m, n, k);
  }
  
}

int main(int argc, char *argv[]) 
{
  // Initialize the MPI environment
  // TODO: Initialize MPI
  MPI_Init(&argc, &argv);


  // Get the rank of the process
  // TODO: Get the rank of the current process
  // Get the number of processes
  // TODO: Get the total number of processes
  int rank, total_process_count;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get the rank of the current process
  MPI_Comm_size(MPI_COMM_WORLD, &total_process_count);  // Get the total number of processes

  SummaOpts opts;
  opts = parse_args(argc, argv);



  timer time_total_program;
  if (rank == 0)
  {    
      timer_start(&time_total_program);
  }
  

  // Broadcast options to all processes
  // TODO: Broadcast the parsed options to all processes
  // Broadcast the options to all processes
  MPI_Bcast(&opts, sizeof(SummaOpts), MPI_BYTE, 0, MPI_COMM_WORLD); //MPI_BYTE: Specifies that the data is being treated as raw bytes. This allows you to send complex data structures.

  // Check if number of processes is a perfect square
  // TODO: Check if the number of processes forms a perfect square grid
  int sqre_root = sqrt(total_process_count); 
  if (sqre_root * sqre_root != total_process_count)
  {
    printf("Error: Number of processes launched should be perfect square");
    return 1;
  }

  int grid_size = sqre_root;  // grid_size is the side-length of the process grid


  // Check if matrix dimensions are compatible with grid size
  if (opts.m % grid_size != 0 || opts.n % grid_size != 0 ||
      opts.k % grid_size != 0) {
    printf("Error: Matrix dimensions must be divisible by grid size (%d)\n",
           grid_size);
    return 1;
  }

  printf("\nMatrix Dimensions:\n");
  printf("A: %d x %d\n", opts.m, opts.k);
  printf("B: %d x %d\n", opts.k, opts.n);
  printf("C: %d x %d\n", opts.m, opts.n);
  printf("Grid size: %d x %d\n", grid_size, grid_size);
  printf("Block size: %d\n", opts.block_size);
  printf("Algorithm: Stationary %c\n", opts.stationary);
  printf("Verbose: %s\n", opts.verbose ? "true" : "false");

  // Call the appropriate SUMMA function based on algorithm variant
  if (opts.stationary == 'a') {
    summa_stationary_a(opts.m, opts.n, opts.k, total_process_count, rank);
  } 
  else if (opts.stationary == 'c') {
    summa_stationary_c(opts.m, opts.n, opts.k, total_process_count, rank);
  } 
  else {
    printf("Error: Unknown stationary option '%c'. Use 'A' or 'C'.\n",
           opts.stationary);
    return 1;
  }


  double total_program_runtime;
  if (rank == 0)
  {
    total_program_runtime = seconds_elapsed(&time_total_program);   // *** Timer End ***

      printf("Total program run time: %f\n", (float) total_program_runtime);     // Delete this
  }


  MPI_Finalize();
  return 0;
}