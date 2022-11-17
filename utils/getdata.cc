#include "getdata.h"

std::vector<double> readFile(std::string file_path)
{
    std::vector<double> file_data;
    std::ifstream in(file_path);
    if (in.is_open())
    {
        while (!in.eof())
        {
            double buf;
            in >> buf;
            file_data.push_back(buf);
        }
    }
    else
    {
        std::cout << "error happened when reading file!" << std::endl;
    }
    return file_data;
}

void getTrainData(std::string file_path, std::vector<double> &x_set, std::vector<double> &y_set)
{
    if (x_set.size() != 0 || y_set.size() != 0)
        return;
    std::vector<double> file_data = readFile(file_path);
    for (size_t i = 0; i < file_data.size(); i += IN_NODE + 1)
    {
        for (size_t j = 0; j < IN_NODE; ++j)
        {
            x_set.push_back(file_data[i + j]);
        }
        y_set.push_back(file_data[i + IN_NODE]);
    }
}

void getTestData(std::string file_path, std::vector<double> &x_set)
{
    if (x_set.size() != 0)
        return;
    x_set = readFile(file_path);
}