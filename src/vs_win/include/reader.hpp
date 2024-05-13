/*
* Functions for reading and processing txt file with floats
* Tomas Goldmann,2024
*/


#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <immintrin.h> 

using namespace std;

void readFloatsFromFile(const string& filename ,__m256 **loc);

