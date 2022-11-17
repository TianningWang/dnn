#ifndef _NETWORK
#define _NETWORK

#include <iostream>
#include <math.h>
#include <string>
#include <vector>
#include <random>
#include "utils/getdata.h"
#include "config.h"

struct Data
{
    std::vector<double> x;
    double y;
    Data();
    Data(const std::vector<double> &x, const double &y) : x(x), y(y) {}
    void printData();
};

inline double sigmoid(double x);

struct Node
{
    double value, bias, delta_bias;
    std::vector<double> weight, delta_weight;
};

class Network
{
public:
    Network();
    void train(const std::vector<Data> &train_data);
    std::vector<double> predict(const std::vector<double> &test_data);
    std::vector<Data> load_train_data(std::string file_path);
    std::vector<double> load_test_data(std::string file_path);
    void showParam();

private:
    void clear_grad();
    void front_propgation();
    void back_propgation(const double y);
    double calculate_loss(const double y);
    void update_param(size_t sample_num);

    Node input_layer[IN_NODE];
    Node hidden_layer[HIDE_NODE];
    Node output_layer;
};

#endif