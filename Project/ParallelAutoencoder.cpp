#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <vector>
#include <tuple>
#include <random>
#include <mpi.h>
#include <stdlib.h> 

#include "autoencoder.h"
#include "utils.h"

using namespace std;

int main(int argc, char **argv) {

	int id; // Process Rank
    int p; // N. of Processes
    double elapsed_time; // Parallel Execution Time
    FILE *fp; // Input Dataset file 
    double *buff_storage; // Input Buffer
    double *recv_storage, *reduce_storage_encoder, *reduce_storage_decoder; // Receive Buffer
    int n, m, t, hidden_dim, epochs; // n array of lenght m of trainset, t array of testset, # of hidden neurons, # of epochs
	double lr, momentum; // hyperparameters
    int *send_count; // array for scatterv
    int *send_disp; // array for scatterv
	int *recv_count; // array for gatherv
    int *recv_disp; // array for gatherv
	double reconstruction_error; 
	double *errors, *total_errors_array;

    MPI_Init(&argc, &argv);

    MPI_Barrier(MPI_COMM_WORLD);
	elapsed_time = -MPI_Wtime();

    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    if (argc != 6 || p > atoi(argv[2]) || p > atoi(argv[5])) {
        if(!id) {
			printf("Usage: ./ParallelAutoencoder\n"
				"- str\tdataset file\n"
				"- int\tnumber of dataset samples (rows in dataset file)\n"
				"- int\tnumber of attributes of each item (columns in dataset file)\n"
				"- str\ttestset file\n"
				"- int\tnumber of testset sample (rows in testset file)\n"
				"- p must be < of n and t\n");
		}
        MPI_Finalize();
        return 0;
    }

    n = atoi(argv[2]);
    m = atoi(argv[3]);
	t = atoi(argv[5]);

	hidden_dim = 20;
	lr = 0.25;
	momentum = 0.9;
	epochs = 300;

	// Process 0 read the training set file, store the array in buff_storage
	if(!id){

		nn::create_random_dataset(fp, argv[1], n, m);
		nn::create_random_dataset(fp, argv[4], t, m);

        if ((fp = fopen(argv[1], "r")) == NULL){
            printf("Error opening the file.");
        }
        else {
            buff_storage = (double *) malloc(n * m * sizeof(double));

            for(int j = 0; j < n * m; j++){
                int b = fscanf(fp, "%lf", &buff_storage[j]);
                //printf("%f\n", buff_storage[j]);
            }
            fclose(fp);
        }
    }

	//Each process performs these functions. Process 0 sends with Scatterv the data portion to the other processes
	nn::create_mixed_xfer_arrays(p, n, m, &send_count, &send_disp);
	
	recv_storage = (double *) malloc(send_count[id] * sizeof(double));

	MPI_Scatterv(buff_storage, send_count, send_disp, MPI_DOUBLE, recv_storage, n*m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//printf("Process %d received %d samples at disp: %d", id, send_count[id]/m, send_disp[id]/m);
	/* for(int k = 0; k < send_count[id]; k++){
        printf(" %f ", recv_storage[k]);
	}
	printf("\n\n"); */

	/*--------------------------------------------TRAINING--------------------------------------------------------*/

	double input[m]; // input store one sample at time and pass it to autoencoder

	Autoencoder *nn = new Autoencoder(m, hidden_dim, lr, momentum);

	for (auto e = 0; e < epochs; e++) {

		for(int i = 0; i < send_count[id]/m; i++) {

			//printf("Process %d input epoch %d: ", id, e);
			for(int j = 0; j < m; j++) {
				input[j] = recv_storage[j+m*i];
				//printf(" %f ", input[j]);
			}
			//printf("\n\n");
			nn->train(input);
		}

		// we can see the progress of training printing the error every 1000 epochs and at the final one
		/* if ((e % 1000 == 0 && e > 0) || e == epochs-1) {
			nn->report(id, e);
		} */
	}

	// Print input and output of each process
	/* printf("Process %d input:\t", id);
	for(int j = 0; j < m; j++) {
		printf("%f ", input[j]);
	}
	printf("\nProcess %d output:\t", id);
	for(int j = 0; j < m; j++) {
		printf("%f ", nn->m_outputValues[j]);
	} */

	// Print encoding and decoding weights matrices 
	/* for(int i = 0; i < hidden_dim * m; i++) {
		printf("%f ", nn->encoderWeights[i]);
	}
	printf("\n\n"); */
	/* for(int i = 0; i < m * hidden_dim; i++) {
		printf("%f ", nn->decoderWeights[i]);
	}
	printf("\n\n"); */

	// Each process allocates these storages and receives from the other processes: 
	// at the end all processes will have the final weight matrices and 
	// load them to the network to carry out the test phase
	reduce_storage_encoder = (double *) malloc(hidden_dim * m * sizeof(double));
	reduce_storage_decoder = (double *) malloc(m * hidden_dim * sizeof(double));
	
	MPI_Allreduce(nn->encoderWeights, reduce_storage_encoder, hidden_dim * m, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(nn->decoderWeights, reduce_storage_decoder, m * hidden_dim, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	// We perform the average of the weights on all processes
	for(int i = 0; i < hidden_dim * m; i++) {
		reduce_storage_encoder[i] = reduce_storage_encoder[i] / p;
		reduce_storage_decoder[i] = reduce_storage_decoder[i] / p;
	}

	// Load the weights into the autoencoder
	for(int i = 0; i < hidden_dim; i++) {
		for(int j = 0; j < m; j++) {
			nn->m_encoderWeights[i][j] = reduce_storage_encoder[j+m*i];
		}
	}

	for(int i = 0; i < m; i++) {
		for(int j = 0; j < hidden_dim; j++) {
			nn->m_decoderWeights[i][j] = reduce_storage_decoder[j+m*i];
		}
	}

	// Deallocate memory
	free(reduce_storage_encoder); 
	free(reduce_storage_decoder);
	
	free(nn->encoderWeights);
	free(nn->decoderWeights);

	/*------------------------------------------------TESTING-----------------------------------------------------------*/
	if(!id){
        if ((fp = fopen(argv[4], "r")) == NULL){
            printf("Error opening the file.");
        }
        else {
            buff_storage = (double *) realloc(buff_storage, t * m * sizeof(double));

            for(int j = 0; j < t * m; j++){
                int b = fscanf(fp, "%lf", &buff_storage[j]);
            }

            fclose(fp);
        }
    }

	nn::create_mixed_xfer_arrays(p, t, m, &send_count, &send_disp);
	
	recv_storage = (double *) realloc(recv_storage, send_count[id] * sizeof(double));

	MPI_Scatterv(buff_storage, send_count, send_disp, MPI_DOUBLE, recv_storage, t*m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	/* printf("Process %d received %d samples", id, send_count[id]/m);
	for(int k = 0; k < send_count[id]; k++){
        printf(" %f ", recv_storage[k]);
	}
	printf("\n\n"); */

	errors = (double * ) malloc((send_count[id]/m) * sizeof(double));

	for(int i = 0; i < send_count[id]/m; i++) {
		//printf("Process %d input epoch %d: ", id, e);
		for(int j = 0; j < m; j++) {
			input[j] = recv_storage[j+m*i];
			//printf(" %f ", input[j]);
		}
		//printf("\n\n");

		nn->test(input);

		// Each process canculates the recostruction errors for its inputs and puts them in "errors" array
		reconstruction_error = 0;

		for(int j = 0; j < m; j++) {
			reconstruction_error += nn::squareError(nn->m_outputValues[j], input[j]);
		}
		reconstruction_error = reconstruction_error/m;

		errors[i] = reconstruction_error;
	}

	// Print the errors array for each process
	/* printf("Process %d: ", id);
	for(int i = 0; i < send_count[id]/m; i++){
		printf("%f ", errors[i]);
	}
	printf("\n\n"); */ 
  
	// Each process sorts the errors array
	qsort(errors, send_count[id]/m, sizeof(double), nn::compare);

	nn::create_arrays (p, m, send_count, send_disp, &recv_count, &recv_disp);

	if(!id){
		total_errors_array =  (double * ) malloc(t * sizeof(double));
	}

	// Each process sends the errors array to process 0, that store them in total_errors_array
	MPI_Gatherv(errors, send_count[id]/m, MPI_DOUBLE, total_errors_array, recv_count, recv_disp, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	if(!id){

		// Print the received array
		/* for(int k = 0; k < t; k++){
			printf(" %f ", total_errors_array[k]);
		}
		printf("\n\n"); */

		vector<double> arr;
		vector<vector<double> > tot_arr;
		int j_tot = 0;

		for(int i = 0; i < p; i++){
		
			for(int j = 0; j < recv_count[i]; j++){

				//printf("%f ", total_errors_array[j_tot]);
				arr.push_back(total_errors_array[j_tot]);
				j_tot++;
			}

			tot_arr.push_back(arr);
			arr.clear();
		}

		// Print the vector of vector tot_arr
		/* for (int i = 0; i < tot_arr.size(); i++) { 
			for (int j = 0; j < tot_arr[i].size(); j++) 
				cout << tot_arr[i][j] << " "; 
			cout << endl; 
    	} */

		// Merge the k ordered array
		vector<double> output = nn::mergeKArrays(tot_arr);
 
    	cout << "The ordered errors array is " << endl;
    	for (auto x : output)
        	cout << x << " ";

		nn::percentile_calculation(output, output.size());
	}

	delete nn;

	if(!id){
		free(buff_storage);
		free(total_errors_array);
	}

	free(recv_storage);
	free(send_count);
	free(send_disp);
	free(recv_count);
    free(recv_disp); 
	free(errors);

	double et;
	MPI_Reduce(&elapsed_time, &et, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	et += MPI_Wtime();

	if (!id)
		printf("\nElapsed time is %f\n", et);

    MPI_Finalize();
	return 0;
}