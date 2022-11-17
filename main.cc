#include <iostream>
#include "network.h"

int main(void)
{
    // train
    Network network;
    // network.showParam();
    std::vector<Data> train_data = network.load_train_data("../data/train.txt");
    network.train(train_data);
    // network.showParam();

    // predict
    std::vector<double> test_data = network.load_test_data("../data/test.txt");
    std::vector<double> rslt = network.predict(test_data);
    for (size_t i = 0; i < rslt.size(); ++i)
    {
        std::cout << "x: " << test_data[2 * i] << " " << test_data[2 * i + 1] << "  y_hat: " << rslt[i] << std::endl;
    }
    return 0;
}