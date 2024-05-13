/*
* Prior boxes for RetinaNet
* Tomas Goldmann,2023
*/


#include <iostream>
#include <vector>
#include <cmath>


#include <immintrin.h> 


class PriorBox {


private:
    std::vector<std::vector<int>> feature_maps;
    std::vector<std::vector<float>> min_sizes;
    std::vector<int> steps;
    bool clip;
    std::vector<int> image_size;
    std::string name;


public:
    PriorBox(std::vector<int> image_size = std::vector<int>(), std::string phase = "train");
    std::vector<std::vector<float>> forward();

};

void decode(__m256 **vec, const std::vector<std::vector<float>>& priors, const std::vector<float>& variances);