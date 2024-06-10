#include "neuralnetwork.h"
#include <fstream>
#include <string>
#include <iostream>
#include <algorithm>
#include <random>

#include "mainwindow.h"
#include "ui_mainwindow.h"

using namespace std;

typedef vector<RowVector*> data_V;

QVector<double> output_noise, input_clear, output_clear, output_NN, error;

void ReadCSV(string filename, vector<RowVector*>& data)
{
    data.clear();
    ifstream file(filename);
    string line, word;
    // determine number of columns in file
    getline(file, line, '\n');
    stringstream ss(line);
    vector<Scalar> parsed_vec;
    while (getline(ss, word, ',')) {
        parsed_vec.push_back(Scalar(stof(&word[0])));
    }
    size_t cols = parsed_vec.size();
    data.push_back(new RowVector(cols));
    for (uint i = 0; i < cols; i++) {
        data.back()->coeffRef(/*1, */i) = parsed_vec[i];
    }

    // read the file
    if (file.is_open()) {
        while (getline(file, line, '\n')) {
            stringstream ss(line);
            data.push_back(new RowVector(/*1, */cols));
            uint i = 0;
            while (getline(ss, word, ',')) {
                data.back()->coeffRef(i) = Scalar(stof(&word[0]));
                i++;
            }
        }
    }
}

double Noise(int low, int high)
{
    return (double)((rand() % ((high + 1) - low) + low)/150.0);
}

void genData(string filename)
{
    ofstream file1(filename);
    double step = 0.01;
    for (double r = 0; r < 10; r += step) {
        double noise = Noise(-100,100);
        double x = r;
        double y = 1 - exp(-x);
        file1 << x << "," << y + noise << endl;
        input_clear.push_back(x);
        output_clear.push_back(y);
        output_noise.push_back(y + noise);
    }
    file1.close();
}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->widget->xAxis->setRange(0, 10);
    ui->widget->yAxis->setRange(-1, 2);

    ui->widget->addGraph();
    ui->widget->addGraph();
    ui->widget->addGraph();
    ui->widget->addGraph();

    NeuralNetwork n({ 1, 5, 1 }); // слои нейронной сети
    data_V datas, datas_start;
    string file_name = "datas.txt";
    genData(file_name);
    ReadCSV(file_name, datas);
    ReadCSV(file_name, datas_start);

    n.output.resize(datas.size());
    n.error.resize(datas.size());
    int epoche = 0;
    int max_epoche = 100;
    while (epoche < max_epoche){
        if (epoche != max_epoche - 1) {
            shuffle(begin(datas), end(datas), default_random_engine {});
            n.train(datas, output_clear);
        }
        else
            n.train(datas_start, output_clear);
        epoche++;
    }

    for (uint r = 0; r < n.output.size(); r++) {
        output_NN.push_back(n.output.at(r));
        error.push_back(n.error.at(r));
    }

    QPen blackDotPen;
    blackDotPen.setColor(QColor(0, 0, 0, 255));
    blackDotPen.setWidthF(3);
    QPen greenDotPen;
    greenDotPen.setColor(QColor(0, 255, 0, 255));
    greenDotPen.setWidthF(3);

    ui->widget->graph(0)->addData(input_clear, output_NN);
    ui->widget->graph(0)->setPen(blackDotPen);
    ui->widget->graph(1)->addData(input_clear, output_clear);
    ui->widget->graph(1)->setPen(greenDotPen);
    ui->widget->graph(2)->addData(input_clear, output_noise);
    ui->widget->graph(2)->setPen(QPen(Qt::red));
    ui->widget->graph(3)->addData(input_clear, error);
    ui->widget->graph(3)->setPen(QPen(Qt::blue));
    ui->widget->replot();

}

MainWindow::~MainWindow()
{
    delete ui;
}

