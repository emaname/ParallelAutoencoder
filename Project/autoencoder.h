#pragma once

class Autoencoder
{
public:
	Autoencoder(int inputDim, int hiddenDim, double learningRate, double momentum);
	~Autoencoder();

	void train(double *data) ;
	void test(double *data) ;

	double *encoderWeights;
	double *decoderWeights;

	double **m_encoderWeights;
	double **m_decoderWeights;

	double *m_inputValues;
	double *m_hiddenValues;
	double *m_outputValues;

	double m_error;
	
	// for debugging purposes
	void fullPrint() const;
	void report(int id, int epoch) const;

private:
	int m_dataDimension; // # of output neurons = # of input neurons
	int m_hiddenDimension;

	double **m_encoderWeightChanges;
	double **m_prevEncoderWeightChanges;
	double **m_decoderWeightChanges;
	double **m_prevDecoderWeightChanges;

	double *m_deltas;

	double m_learningRate;
	double m_momentum;

	void feedforward() ;
	void backpropagate() const;

};