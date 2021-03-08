#include <random>
#include <algorithm>
#include <iostream>
#include <bits/stdc++.h>

#define BLOCK_LOW(id,p,n)  ((id)*(n)/(p))
#define BLOCK_HIGH(id,p,n) (BLOCK_LOW((id)+1,p,n)-1)
#define BLOCK_SIZE(id,p,n) (BLOCK_HIGH(id,p,n)-BLOCK_LOW(id,p,n)+1)
#define BLOCK_OWNER(j,p,n) (((p)*((j)+1)-1)/(n))

#define mpitype MPI_DOUBLE

using namespace std;

typedef pair<double, pair<double, double> > ppi;

namespace nn {

	double *random(size_t elementSize) {
		double *result = new double[elementSize];
		for (size_t i = 0; i < elementSize; i++) {
			result[i] = ((double)rand() / (RAND_MAX));
			// result[i] = -1 + 2 * ((float)rand()) / RAND_MAX;
		}
		return result;
	}

	double squareError(double d1, double d2) {
		return pow((d1 - d2), 2);
	}

	double sigmoid(double d) {
		return 1.0 / (1.0 + exp(-d));
	}

	double sigmoidDerivation(double d) {
		return d * (1.0 - d);
	}

	double relu(double d) {
		return std::fmax(0, d);
	}

	double reluDerivation(double d) {
		return d >= 0.0 ? 0.0 : 1.0;
	}

	// Function used in qsort
	int compare(const void* a, const void* b) {
		double va = *(const double*) a;
		double vb = *(const double*) b;
		return (va > vb) - (va < vb);
	}

	// Create send_count and send_disp array for Scatterv
	void create_mixed_xfer_arrays (int p, int n, int m, int **count, int **disp) {

		*count = (int*)malloc(p * sizeof(int));
		*disp = (int*)malloc(p * sizeof(int));
		(*count)[0] = BLOCK_SIZE(0,p,n) * m;
		(*disp)[0] = 0;
		for (int i = 1; i < p; i++) {
			(*disp)[i] = (*disp)[i-1] + (*count)[i-1];
			(*count)[i] = BLOCK_SIZE(i,p,n) * m;
		}
	}

	// Create recv_count and recv_disp array for Gatherv
	void create_arrays (int p, int m, int *send_count, int *send_disp, int **recv_count, int **recv_disp) {

		*recv_count = (int*)malloc(p * sizeof(int));
		*recv_disp = (int*)malloc(p * sizeof(int));
		
		for (int i = 0; i < p; i++) {
			(*recv_disp)[i] = send_disp[i]/m;
			(*recv_count)[i] =send_count[i]/m;
		}
	}

	// Normalize data in [0,1]
	void normalize_data (int n, int m, double *dataset) {

		double max = numeric_limits<double>::max();
		double min = numeric_limits<double>::min();
		double *mins = (double*)malloc(m * sizeof(double));
		double *maxes = (double*)malloc(m * sizeof(double));

		for(int i = 0; i < m; i++) {
			mins[i] = max;
			maxes[i] = min;
		}

		for(int i = 0; i < m; i++) {
			for(int j = 0; j < n; j++) {
				if(dataset[j*m+i] > maxes[i])
					maxes[i] = dataset[j*m+i];
				if(dataset[j*m+i] < mins[i])
					mins[i] = dataset[j*m+i];
			}
		}

		for(int i = 0; i < n; i++) {
			for(int j = 0; j < m; j++) {
				dataset[j+m*i] = (dataset[j+m*i] - mins[j]) / (maxes[j] - mins[j]);
			}
		}

		free(mins);
		free(maxes);
	}

	// Create dataset with Gaussian distribution(mean, sigma)
	void create_random_dataset (FILE *fp, char *file, int n, int m) {

		default_random_engine generator;
  		normal_distribution<double> distribution(0,1);

		double *dataset = (double*)malloc(n * m * sizeof(double));

		for(int i = 0; i < n * m; i++) {
			dataset[i] = (double)distribution(generator);
		}
	
		normalize_data(n, m, dataset);

		if ((fp = fopen(file, "w")) == NULL){
			printf("Error opening the file.");
		}
		for(int i = 0; i < n; i++) {
			for (int j = 0; j < m; ++j) {
				fprintf(fp, "%f ", dataset[j+m*i]);
			}
			fprintf(fp, "\n");
		}
		free(dataset);
		fclose(fp);
	}

	// This function takes an array of arrays as an
	// argument and all arrays are assumed to be
	// sorted. It merges them together and prints
	// the final sorted output.
	vector<double> mergeKArrays(vector<vector<double> > arr) {

		vector<double> output;
	
		// Create a min heap with k heap nodes. Every
		// heap node has first element of an array
		priority_queue<ppi, vector<ppi>, greater<ppi> > pq;
	
		for (int i = 0; i < arr.size(); i++)
			pq.push({ arr[i][0], { i, 0 } });
	
		// Now one by one get the minimum element
		// from min heap and replace it with next
		// element of its array
		while (pq.empty() == false) {
			ppi curr = pq.top();
			pq.pop();
	
			// i ==> Array Number
			// j ==> Index in the array number
			double i = curr.second.first;
			int j = curr.second.second;
	
			output.insert(output.begin(), curr.first);
	
			// The next element belongs to same array as
			// current.
			if (j + 1 < arr[i].size())
				pq.push({ arr[i][j + 1], { i, j + 1 } });
		}
	
		return output;
	}

	void percentile_calculation(vector<double> output, int size){
		
		for(int i = 0; i < size; i++) { 
        	
			int count=0;

			for(int j = 0; j < size; j++) {
				
				if(output[i] > output[j])
					count=count+1;
			}
			
			int percent = (count*100)/(size-1);

			if(percent >= 90) {
				cout<<"\nThe value "<<output[i]<<" is an outlier (percentile = "<<percent<<")";
			}	
    	}    
	}
};