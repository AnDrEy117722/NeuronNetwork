#include "neuralnetwork.h"

// constructor of neural network class
NeuralNetwork::NeuralNetwork(vector<uint> topology, Scalar learningRate)
{
    var_bool = true;

    this->topology = topology;
    this->learningRate = learningRate;
    for (uint i = 0; i < topology.size(); i++) {
        // initialize neuron layers
        if (i == topology.size() - 1)
            neuronLayers.push_back(new RowVector(topology[i]));
        else
            neuronLayers.push_back(new RowVector(topology[i] + 1));

        // initialize cache and delta vectors
        cacheLayers.push_back(new RowVector(neuronLayers.back()->size()));
        deltas.push_back(new RowVector(neuronLayers.back()->size()));

        // vector.back() gives the handle to recently added element
        // coeffRef gives the reference of value at that place
        // (using this as we are using pointers here)
        if (i != topology.size() - 1) {
            neuronLayers.back()->coeffRef(topology[i]) = 1.0;
            cacheLayers.back()->coeffRef(topology[i]) = 1.0;
        }

        // initialize weights matrix
        if (i > 0) {
            if (i != topology.size() - 1) {
                weights.push_back(new Matrix(topology[i - 1] + 1, topology[i] + 1));
                weights.back()->setRandom();
                weights.back()->col(topology[i]).setZero();
                weights.back()->coeffRef(topology[i - 1], topology[i]) = 1.0;
            }
            else {
                weights.push_back(new Matrix(topology[i - 1] + 1, topology[i]));
                weights.back()->setRandom();
            }
        }
    }
};

Scalar activationFunction(Scalar x)
{
    return 1.0/(1.0 + exp(-x)); //линейная функция активации
//    return tanhf(x);
}

Scalar activationFunction_last(Scalar x)
{
    return x; //линейная функция активации
}

Scalar activationFunctionDerivative(Scalar x)
{
//    return 1 - tanhf(x) * tanhf(x)/*1*/;
    return activationFunction(x)*(1 - activationFunction(x));
}

Scalar activationFunctionDerivative_last(Scalar x)
{
//    return 1 - tanhf(x) * tanhf(x)/*1*/;
    return 1;
}

void NeuralNetwork::propagateForward(RowVector& input)
{
    // set the input to input layer
    // block returns a part of the given vector or matrix
    // block takes 4 arguments : startRow, startCol, blockRows, blockCols

    neuronLayers.front()->block(0, 0, 1, neuronLayers.front()->size() - 1) = input;

    // propagate the data forward and then
    // apply the activation function to your network
    // unaryExpr applies the given function to all elements of CURRENT_LAYER
    for (uint i = 1; i < topology.size(); i++) {
        // already explained above
        (*cacheLayers[i]) = (*neuronLayers[i - 1]) * (*weights[i - 1]);
          if (i != topology.size() - 1)
              neuronLayers[i]->block(0, 0, 1, topology[i]) =
                      cacheLayers[i]->block(0, 0, 1, topology[i]).unaryExpr(function<Scalar(Scalar)>(activationFunction));
          else
              neuronLayers[i]->block(0, 0, 1, topology[i]) =
                      cacheLayers[i]->block(0, 0, 1, topology[i]).unaryExpr(function<Scalar(Scalar)>(activationFunction_last));
    }
}

void NeuralNetwork::calcErrors(RowVector& output)
{
    // calculate the errors made by neurons of last layer
    (*deltas.back()) = output - (*neuronLayers.back());

    // error calculation of hidden layers is different
    // we will begin by the last hidden layer
    // and we will continue till the first hidden layer
    for (size_t i = topology.size() - 2; i > 0; i--) {
        (*deltas[i]) = (*deltas[i + 1]) * (weights[i]->transpose());
    }
}

//используется стахостический градиентный спуск
void NeuralNetwork::updateWeights()
{
    // topology.size()-1 = weights.size()
    for (uint i = 0; i < topology.size() - 1; i++) {
        // in this loop we are iterating over the different layers (from first hidden to output layer)
        // if this layer is the output layer, there is no bias neuron there, number of neurons specified = number of cols
        // if this layer not the output layer, there is a bias neuron and number of neurons specified = number of cols -1
        if (i != topology.size() - 2) {
            for (uint c = 0; c < weights[i]->cols() - 1; c++) {
                for (uint r = 0; r < weights[i]->rows(); r++) {
                    weights[i]->coeffRef(r, c) += learningRate * deltas[i + 1]->coeffRef(c) * activationFunctionDerivative(cacheLayers[i + 1]->coeffRef(c)) * neuronLayers[i]->coeffRef(r);
                }
            }
        }
        else {
            for (uint c = 0; c < weights[i]->cols(); c++) {
                for (uint r = 0; r < weights[i]->rows(); r++) {
                    weights[i]->coeffRef(r, c) += learningRate * deltas[i + 1]->coeffRef(c) * activationFunctionDerivative_last(cacheLayers[i + 1]->coeffRef(c)) * neuronLayers[i]->coeffRef(r);
                }
            }
        }
    }
}

void NeuralNetwork::propagateBackward(RowVector& output)
{
    calcErrors(output);
    updateWeights();
}

void NeuralNetwork::train(vector<RowVector*> data, QVector<double> y_clear)
{
    RowVector* input = new RowVector(1);
    RowVector* output_new = new RowVector(1);
    for (uint i = 0; i < data.size(); i++) {
        input->coeffRef(0) = data[i]->coeffRef(0);
        output_new->coeffRef(0) = data[i]->coeffRef(1);
        propagateForward(*input);
        output.at(i) = neuronLayers.back()->coeffRef(0);
        error.at(i) = (neuronLayers.back()->coeffRef(0) - y_clear.at(i));
        bool condition = i > data.size() - 1;
//        bool condition = i < 1;
//        bool condition = i > 0;
        if (condition) {
            cout << "Input to neural network is : " << data[i]->coeffRef(0) << endl;
            cout << "Expected output is : " << data[i]->coeffRef(1) << endl;
            cout << "Output produced is : " << *neuronLayers.back() << endl;
        }
//        if (i < data.size() / 2)
//            learningRate += 0.00001f;
//        else
//            learningRate -= 0.00001f;
        propagateBackward(*output_new);
        if (condition) {
            cout << "MSE : " << sqrt((*deltas.back()).dot((*deltas.back())) / deltas.back()->size()) << endl;
        }

    }

    delete(input);
    delete(output_new);


}

RowVector* NeuralNetwork::check(RowVector& input_data){

    neuronLayers.front()->block(0, 0, 1, neuronLayers.front()->size() - 1) = input_data;

    for (uint i = 1; i < topology.size(); i++) {
        (*neuronLayers[i]) = (*neuronLayers[i - 1]) * (*weights[i - 1]);
        if (i != topology.size() - 1)
            neuronLayers[i]->block(0, 0, 1, topology[i]).unaryExpr(ptr_fun(activationFunction));
        else
            neuronLayers[i]->block(0, 0, 1, topology[i]).unaryExpr(ptr_fun(activationFunction_last));
    }

    return neuronLayers.back();

}


