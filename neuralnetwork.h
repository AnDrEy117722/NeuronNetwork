#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H


// NeuralNetwork.hpp
#include <Eigen/Eigen>
#include <iostream>
#include <vector>

#include <QVector>

// use typedefs for future ease for changing data types like : float to double
typedef float Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;
typedef uint32_t uint;

using namespace std;

// neural network implementation class!
class NeuralNetwork {
public:
    // constructor
    NeuralNetwork(vector<uint> topology, Scalar learningRate = Scalar(0.001));

    // function for forward propagation of data
    void propagateForward(RowVector& input);

    // function for backward propagation of errors made by neurons
    void propagateBackward(RowVector& output);

    // function to calculate errors made by neurons in each layer
    void calcErrors(RowVector& output);

    // function to update the weights of connections
    void updateWeights();

    // function to train the neural network give an array of data points
//    void train(vector<RowVector*> input_data, vector<RowVector*> output_data);
    void train(vector<RowVector*> data, QVector<double> y_clear);

    RowVector* check(RowVector& input_data);

    // storage objects for working of neural network
    /*
        use pointers when using vector<Class> as vector<Class> calls destructor of
        Class as soon as it is pushed back! when we use pointers it can't do that, besides
        it also makes our neural network class less heavy!! It would be nice if you can use
        smart pointers instead of usual ones like this
        */
    vector<Scalar> output;
    vector<Scalar> error;
    vector<RowVector*> neuronLayers; // stores the different layers of out network
    vector<RowVector*> cacheLayers; // stores the unactivated (activation fn not yet applied) values of layers
    vector<RowVector*> deltas; // stores the error contribution of each neurons
    vector<Matrix*> weights; // the connection weights itself
    vector<uint> topology;
    Scalar learningRate;

    bool var_bool;
};


#endif // NEURALNETWORK_H
