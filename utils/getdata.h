#ifndef _GET_DATA
#define _GET_DATA
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "../config.h"

/*
function: read file basic function
input:  file path  type:  std::string
output: file data  type:  std::vector<double>
*/
std::vector<double> readFile(std::string file_path);

/*
function: get training data
input:
output:
*/
void getTrainData(std::string file_path, std::vector<double> &x_set, std::vector<double> &y_set);

void getTestData(std::string file_path, std::vector<double> &x_set);

#endif