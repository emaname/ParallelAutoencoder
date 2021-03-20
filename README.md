# ParallelAutoencoder

Implementation of a parallel autoencoder, using C/C++ and MPI. 
This implementation uses the autoencoder.cpp of the following github repository: https://github.com/turkdogan/autoencoder and is based on the parallel design proposed by the work of Yunlong Ma, Peng Zhang, Yanan Cao, Li Guo, "Parallel  Auto-encoder  for  Efficient  Outlier  Detection",  2013  IEEE  International Conference on Big Data [link to pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6691791&casa_token=1Kh56D7aCwIAAAAA:tZLM44rxbmLje_24m-XV_XeTePgPI-1rde66aB1frfD_VhZTFdF79I0PGkaHFLazeSt1LLEQazrMkA&tag=1)

How to run:
- mpiCC autoencoder.cpp -c -lm
- mpiCC utils.cpp -c -lm
- mpiCC  autoencoder.o  utils.o  -O3  ParallelAutoencoder.cpp  -o  ParallelAutoencoder -lm
- mpirun -np <number of process> ./ParallelAutoencoder <trainingset-filename.txt> <number of training samples> <length of samples><testset-filename.txt> <number of testing samples>
