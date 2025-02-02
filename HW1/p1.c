#include <mpi.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include <stdlib.h>

int main(int argc, char** argv) 
{
    // Initialize the MPI environment.
    MPI_Init(NULL, NULL);

    // MPI Process variables
    int process_number;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_number);
    int num_of_processes;
    MPI_Comm_size(MPI_COMM_WORLD, &num_of_processes);


    // Timer variables
    struct timeval end_time;
    struct timeval start_time;
    double elapsed;

    // This array will be cleared and reused again and again for each message size
    // Only process-0 will handle the lantency_values[10] array.
    // It will use it for calculating metric outputs. latency_values[i] = Latency value calculated between pair-0 in iteration-i 
    double* latency_values = NULL;  //Initialising outside to prevent scope errors
    if (process_number == 0)
    {
        latency_values = (double*) malloc(10 * sizeof(double));
    }

    
    // Iterate message size from 16, 32, .., 2048. NOTE: 16 is a dummy warmup iteration used to prevent unnaturally high round times in the 32 iteration
    for (int message_size_in_kb = 16; message_size_in_kb <= 2048; message_size_in_kb = message_size_in_kb * 2)
    {
        // Make process-0 clear out the entire latency_values[10] array at the start of each message_size iteration
        if (process_number == 0)
        {
            for (int i = 0; i < 10; i++) 
            {
                latency_values[i] = 0.0;
            }
        }

        // Number of ints in the message: Convert kb to bytes and then divide by size of int in bytes
        int number_of_ints_in_message = (message_size_in_kb * 1024) / sizeof(int);   

        // Create an integer array holding message_size integers.
        // Every single process needs to have this message[] array. 
        // ( The sender will store and send data from message[] array. The reciever will have this array initialised, and use it as a buffer to store the data sent from the sender )
        int* message = (int *) malloc(number_of_ints_in_message * sizeof(int));
        
        // Other variables used for calculating metrics for each message size iteration
        double avg_latency = 0;
        double std_dev = 0;
        double squared_diff_sum = 0;

        //10 iterations of a given message size to calulate average and std-dev
        for (int i=0; i<10; i++)
        {
            // Even processes send to and receive from the next process number
            if (process_number % 2 == 0)    
            {
                // We measure the time only for the first pair (process-0 and process-1) in every iteration
                // Start timer when the process-0 sends message to process-1
                if (process_number == 0)
                {
                    gettimeofday(&start_time, NULL);    
                }

                // Send and receive from the next process number (of the same iteration, paired by tag number)
                MPI_Send(message, number_of_ints_in_message, MPI_INT, process_number + 1, i, MPI_COMM_WORLD);                        // tag set as i to avoid conflicts across iterations
                MPI_Recv(message, number_of_ints_in_message, MPI_INT, process_number + 1, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);  // Round trip receive


                // Process-0 segment: Calculations for first pair: average and std-dev calculation among all iterations done by process-0 alone
                // Stop measuring time when process-0 receives back from process-1 (Round trip completed for current iteration)
                // Calculate the sum of all the latencies only for the first pair
                if (process_number == 0)
                {
                    //Stop timer
                    gettimeofday(&end_time, NULL);
                    elapsed = (end_time.tv_sec - start_time.tv_sec) + ((end_time.tv_usec - start_time.tv_usec) / 1000000.0);

                    // printf("Iteration number: %d", i);
                    // printf("\nRound Trip Latency: %fs\n\n", elapsed);
                    
                    //Note down the latency value for first pair in the iteration-i
                    latency_values[i] = elapsed;
                }
            }
            
            // Odd processes receive from and send to previous process (of the same iteration (of the same iteration, paired by tag number)
            if (process_number % 2 == 1) 
            {
                // Receive from and send to the previous process number
                MPI_Recv(message, number_of_ints_in_message, MPI_INT, process_number - 1, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(message, number_of_ints_in_message, MPI_INT, process_number - 1, i, MPI_COMM_WORLD);
            }
        }

        // Common segment for all 10 iterations of a given message size. Below segment runs once for each message size.
        // Make the first process print the average and std-dev of the latency across the 10 iterations.
        if (process_number == 0)
        {
            // Calc average latency. (Skip the 0th iteration)
            for (int i=1; i<10; i++)
            {
                avg_latency = avg_latency + latency_values[i]; 
            }
            avg_latency = avg_latency / 10;

            // Calc std deviation. (Skip the 0th iteration)
            squared_diff_sum = 0.0;
            for (int i = 1; i < 10; i++) 
            {
                squared_diff_sum += (latency_values[i] - avg_latency) * (latency_values[i] - avg_latency);
            }
            std_dev = sqrt(squared_diff_sum / 10);

            //Print the metrics for the current message_size
            if (message_size_in_kb != 16)
            {
                printf("%d ", message_size_in_kb);
                printf("%f ", avg_latency);
                printf ("%f\n", std_dev);
            }

            // printf("Message Size : %d kb\n", message_size_in_kb);
            // printf("Array calculated Average Round Trip Latency: %fs\n", avg_latency);
            // printf ("The standard deviation is: %f \n\n", std_dev);
        }

        free(message);   // We create a new message in every iteration of message size, so free message at the end of all 10 iterations of a message size
    }

    // Latency values array will be cleared out and reused across all the different message sizes
    // So free it only after all the message size iterations are complete
    if (process_number == 0)
    {
        free(latency_values);
    }

    MPI_Finalize();
}

