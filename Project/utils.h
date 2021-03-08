#pragma once

using namespace std;

namespace nn {

	double *random(size_t elementSize);

	double squareError(double d1, double d2);

	double sigmoid(double d);

	double sigmoidDerivation(double d);

	double relu(double d);

	double reluDerivation(double d);

	int compare(const void * a, const void * b);

	void create_mixed_xfer_arrays(int p, int n, int m, int **count, int **disp);

	void create_arrays(int p, int m, int *send_count, int *send_disp, int **recv_count, int **recv_disp);

	void normalize_data(int n, int m, double *dataset);
	
	void create_random_dataset(FILE *fp, char *file, int n, int m);

	vector<double> mergeKArrays(vector<vector<double> > arr);

	void percentile_calculation(vector<double> output, int size);
};

