#include "network.h"

namespace NetworkParam
{
    double learning_rate = 0.8;
    double early_stopping_threshold = 1e-4;
    size_t epoch_time = 1e6;
};

void Data::printData()
{
    std::cout << "data x: ";
    for (auto iter = x.begin(); iter != x.end(); ++iter)
    {
        std::cout << *iter << " ";
    }
    std::cout << std::endl
              << "data y: " << y << std::endl;
}

Network::Network()
{
    std::mt19937 rand;
    rand.seed(std::random_device()());
    std::uniform_real_distribution<double> distribution(-1, 1);

    // init input layer
    for (size_t i = 0; i < IN_NODE; ++i)
    {
        for (size_t j = 0; j < HIDE_NODE; ++j)
        {
            input_layer[i].weight.push_back(distribution(rand));
            input_layer[i].delta_weight.push_back(0.f);
        }
    }

    // init hidden layer
    for (size_t j = 0; j < HIDE_NODE; ++j)
    {
        hidden_layer[j].bias = distribution(rand);
        hidden_layer[j].delta_bias = 0.f;
        hidden_layer[j].weight.push_back(distribution(rand));
        hidden_layer[j].delta_weight.push_back(0.f);
    }

    // init output layer
    output_layer.bias = distribution(rand);
    output_layer.delta_bias = 0.f;
}

void Network::showParam()
{
    std::cout << "input layer:" << std::endl;
    for (size_t i = 0; i < IN_NODE; ++i)
    {
        std::cout << "node " << i << " weight:  ";
        for (auto iter = input_layer[i].weight.begin(); iter != input_layer[i].weight.end(); ++iter)
        {
            std::cout << *iter << " ";
        }
        std::cout << std::endl;
        std::cout << "delta weight: ";
        for (auto iter = input_layer[i].delta_weight.begin(); iter != input_layer[i].delta_weight.end(); ++iter)
        {
            std::cout << *iter << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "hidden layer:" << std::endl;
    for (size_t j = 0; j < HIDE_NODE; ++j)
    {
        std::cout << "node " << j << " weight: ";
        for (auto iter = hidden_layer[j].weight.begin(); iter != hidden_layer[j].weight.end(); ++iter)
        {
            std::cout << *iter << " ";
        }
        std::cout << std::endl;
        std::cout << "delta weight: ";
        for (auto iter = hidden_layer[j].delta_weight.begin(); iter != hidden_layer[j].delta_weight.end(); ++iter)
        {
            std::cout << *iter << " ";
        }
        std::cout << std::endl;
        std::cout << "node " << j << " bias: " << hidden_layer[j].bias << " delta bias: " << hidden_layer[j].delta_bias << std::endl;
    }
    std::cout << "output layer:" << std::endl;
    std::cout << "bias: " << output_layer.bias << "delta bias: " << output_layer.delta_bias << std::endl;
}

void Network::clear_grad()
{
    // clear input layer weight
    for (size_t i = 0; i < IN_NODE; ++i)
    {
        input_layer[i].delta_weight.assign(input_layer[i].delta_weight.size(), 0.f);
    }

    // clear hidden layer weight and bias
    for (size_t j = 0; j < HIDE_NODE; ++j)
    {
        hidden_layer[j].delta_bias = 0.f;
        hidden_layer[j].delta_weight.assign(hidden_layer[j].delta_weight.size(), 0.f);
    }

    // clear output layer bias
    output_layer.delta_bias = 0.f;
}

std::vector<Data> Network::load_train_data(std::string file_path)
{
    std::vector<double> x_set, y_set;
    std::vector<Data> training_set;
    getTrainData(file_path, x_set, y_set);
    for (size_t i = 0; i < y_set.size(); ++i)
    {
        std::vector<double> x;
        x.push_back(x_set[2 * i]);
        x.push_back(x_set[2 * i + 1]);
        Data tmp_data(x, y_set[i]);
        // tmp_data.printData();
        training_set.push_back(tmp_data);
    }
    // std::cout << training_set.size() << std::endl;
    std::cout << "load training data" << std::endl;
    return training_set;
}

std::vector<double> Network::load_test_data(std::string file_path)
{
    std::vector<double> test_set = {};
    getTestData(file_path, test_set);
    return test_set;
}

// cal value(y) of each node
void Network::front_propgation()
{
    // input layer ==> hidden layer
    for (size_t j = 0; j < HIDE_NODE; ++j)
    {
        double sum = 0;
        for (size_t i = 0; i < IN_NODE; ++i)
        {
            sum += input_layer[i].value * input_layer[i].weight[j];
        }
        sum -= hidden_layer[j].bias;
        hidden_layer[j].value = sigmoid(sum);
    }

    // hidden layer ==> output layer
    double sum = 0;
    for (size_t j = 0; j < HIDE_NODE; ++j)
    {
        sum += hidden_layer[j].value * hidden_layer[j].weight[0];
    }
    sum -= output_layer.bias;
    output_layer.value = sigmoid(sum);
}

double Network::calculate_loss(const double y)
{
    return (std::fabs(output_layer.value - y)) * (std::fabs(output_layer.value - y));
}

// cal delta w/b of each node
// add all delta in one loop of all training data
void Network::back_propgation(const double y)
{
    double y_hat = output_layer.value;
    // update output layer delta bias
    // += means we add all samples delta in back-propgation and then get the average of all samples' delta value
    output_layer.delta_bias += -(y - y_hat) * y_hat * (1.0 - y_hat);

    // update hidden layer delta weight
    for (size_t j = 0; j < HIDE_NODE; ++j)
    {
        hidden_layer[j].delta_weight[0] += (y - y_hat) * y_hat * (1.0 - y_hat) * hidden_layer[j].value;
    }

    // update hidden layer delta biase
    for (size_t j = 0; j < HIDE_NODE; ++j)
    {
        hidden_layer[j].delta_bias += -(y - y_hat) * y_hat * (1.0 - y_hat) * hidden_layer[j].weight[0] * hidden_layer[j].value * (1.0 - hidden_layer[j].value);
    }

    // update input layer delta bias
    for (size_t i = 0; i < IN_NODE; ++i)
    {
        for (size_t j = 0; j < HIDE_NODE; ++j)
        {
            input_layer[i].delta_weight[j] += (y - y_hat) * y_hat * (1.0 - y_hat) * hidden_layer[j].weight[0] * hidden_layer[j].value * (1.0 - hidden_layer[j].value) * input_layer[i].value;
        }
    }
}

void Network::update_param(size_t data_num)
{
    // update input layer w
    for (size_t i = 0; i < IN_NODE; ++i)
    {
        for (size_t j = 0; j < HIDE_NODE; ++j)
        {
            input_layer[i].weight[j] += NetworkParam::learning_rate * input_layer[i].delta_weight[j] / double(data_num);
        }
    }

    // update hidden layer b
    for (size_t j = 0; j < HIDE_NODE; ++j)
    {
        hidden_layer[j].bias += NetworkParam::learning_rate * hidden_layer[j].delta_bias / double(data_num);
        hidden_layer[j].weight[0] += NetworkParam::learning_rate * hidden_layer[j].delta_weight[0] / double(data_num);
    }

    // update output layer bias
    output_layer.bias += NetworkParam::learning_rate * output_layer.delta_bias / double(data_num);
}

void Network::train(const std::vector<Data> &train_data)
{
    for (size_t epoch = 0; epoch <= NetworkParam::epoch_time; ++epoch)
    {
        clear_grad();
        double max_loss = 0.f;

        for (auto iter = train_data.begin(); iter != train_data.end(); ++iter)
        {
            // set x value
            for (size_t i = 0; i < IN_NODE; ++i)
                input_layer[i].value = iter->x[i];
            front_propgation();
            double loss = calculate_loss(iter->y);
            max_loss = std::max(max_loss, loss);
            back_propgation(iter->y);
        }

        // early stopping
        if (max_loss < NetworkParam::early_stopping_threshold)
        {
            std::cout << "early stopping at " << epoch << " epochs" << std::endl;
            std::cout << "final maximum loss: " << max_loss << std::endl;
            return;
        }
        else
        {
            if (epoch % 1000 == 0)
            {
                std::cout << "epochs: " << epoch << " - max loss: " << max_loss << std::endl;
            }
        }

        // update weight and bias
        update_param(train_data.size());
    }
}

std::vector<double> Network::predict(const std::vector<double> &test_data)
{
    std::vector<double> rslt = {};
    for (size_t i = 0; i < test_data.size(); i += IN_NODE)
    {
        for (size_t j = 0; j < IN_NODE; ++j)
        {
            input_layer[j].value = test_data[i + j];
        }
        front_propgation();
        rslt.push_back(output_layer.value);
    }
    return rslt;
}

double sigmoid(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}