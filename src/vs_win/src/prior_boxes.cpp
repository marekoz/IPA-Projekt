/*
* Prior boxes for RetinaNet
* Tomas Goldmann,2023
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <algorithm>
#include "prior_boxes.hpp"
#include <fstream>
#include <immintrin.h>

PriorBox::PriorBox(std::vector<int> image_size, std::string phase)
{

    std::map<std::string, std::vector<std::vector<float>>> cfg;
    this->min_sizes = {{16.0f, 32.0f}, {64.0f, 128.0f}, {256.0f, 512.0f}};
    this->image_size = {image_size[1], image_size[0]};
    this->steps = {8, 16, 32};

    for (const float step : steps)
    {
        this->feature_maps.push_back({(int)std::ceil(image_size[0] / step), (int)std::ceil(image_size[1] / step)});
    }
}

std::vector<std::vector<float>> PriorBox::forward()
{

    std::vector<float> anchors;
    for (size_t k = 0; k < this->feature_maps.size(); k++)
    {
        const auto &f = this->feature_maps[k];
        const auto &min_sizes = this->min_sizes[k];

        for (int i = 0; i < f[1]; i++)
        {
            for (int j = 0; j < f[0]; j++)
            {
                for (const auto &min_size : min_sizes)
                {
                    float s_kx = min_size / this->image_size[1];
                    float s_ky = min_size / this->image_size[0];
                    std::vector<float> dense_cx = {static_cast<float>(j + 0.5) * this->steps[k] / this->image_size[1]};
                    std::vector<float> dense_cy = {static_cast<float>(i + 0.5) * this->steps[k] / this->image_size[0]};

                    for (const auto &cy : dense_cy)
                    {
                        for (const auto &cx : dense_cx)
                        {

                            anchors.push_back(cx);
                            anchors.push_back(cy);
                            anchors.push_back(s_kx);
                            anchors.push_back(s_ky);
                        }
                    }
                }
            }
        }
    }

    std::vector<std::vector<float>> output;

    size_t num_anchors = anchors.size() / 4;
    for (size_t i = 0; i < num_anchors; i++)
    {
        for (size_t j = 0; j < 4; j++)
        {
            anchors[i * 4 + j] = std::min(std::max(anchors[i * 4 + j], 0.0f), 1.0f);
        }
        output.push_back({anchors[i * 4], anchors[i * 4 + 1], anchors[i * 4 + 2], anchors[i * 4 + 3]});
    }
    //std::cout << "len  " << output.size() << std::endl;

    return output;
}

// Taylor expansion for exponential function
// ex = 1 +x/1! + x2/2! + x3/3! + .... source: https://www.efunda.com/math/taylor_series/exponential.cfm
__m256 exp_avx(__m256 x) {
    __m256 result = _mm256_set1_ps(1.0f);
    __m256 term = _mm256_set1_ps(1.0f); // 1

    for (int i = 1; i < 10; ++i) {
        term = _mm256_mul_ps(term, x); // x, *x = x2, x3 ...
        term = _mm256_div_ps(term, _mm256_set1_ps(i)); // x/1, x2/(2*1), x3/(3*2*1)...
        result = _mm256_add_ps(result, term);  // 1 + x/1, 1 + x/1 + x2/2!, ...
    }
    return result;
}


const int AVX_ROWS_COUNT = 1575;
void decode(__m256 **vec, const std::vector<std::vector<float>>& priors, const std::vector<float>& variances)
{
    __m256 two_ps = _mm256_set1_ps(2.0f);
    __m256 vars_x_y = _mm256_set1_ps(variances[0]);
    __m256 vars_w_h = _mm256_set1_ps(variances[1]);
    for (size_t i = 0; i < AVX_ROWS_COUNT; i++) // decoded x
    {
        __m256 priors_x = _mm256_set_ps(priors[i*8+7][0], priors[i*8+6][0], priors[i*8+5][0], priors[i*8+4][0], priors[i*8+3][0], priors[i*8+2][0], priors[i*8+1][0], priors[i*8][0]);
        __m256 priors_y = _mm256_set_ps(priors[i*8+7][1], priors[i*8+6][1], priors[i*8+5][1], priors[i*8+4][1], priors[i*8+3][1], priors[i*8+2][1], priors[i*8+1][1], priors[i*8][1]);
        __m256 priors_w = _mm256_set_ps(priors[i*8+7][2], priors[i*8+6][2], priors[i*8+5][2], priors[i*8+4][2], priors[i*8+3][2], priors[i*8+2][2], priors[i*8+1][2], priors[i*8][2]);
        __m256 priors_h = _mm256_set_ps(priors[i*8+7][3], priors[i*8+6][3], priors[i*8+5][3], priors[i*8+4][3], priors[i*8+3][3], priors[i*8+2][3], priors[i*8+1][3], priors[i*8][3]);

        //float decoded_w = prior_w * std::exp(loc_pred[2] * var_w); 
        __m256 decoded_w = _mm256_mul_ps(vec[2][i], vars_w_h);

        decoded_w = exp_avx(decoded_w);
        decoded_w = _mm256_mul_ps(decoded_w, priors_w);
        decoded_w = _mm256_div_ps(decoded_w, two_ps);

        //float decoded_h = prior_h * std::exp(loc_pred[3] * var_h)
        __m256 decoded_h = _mm256_mul_ps(vec[3][i], vars_w_h);
        decoded_h = exp_avx(decoded_h);
        decoded_h = _mm256_mul_ps(decoded_h, priors_h);
        decoded_h = _mm256_div_ps(decoded_h, two_ps);



        //float decoded_x = prior_x + loc_pred[0] * var_x * prior_w;
        vec[0][i] = _mm256_mul_ps(vec[0][i], vars_x_y);
        vec[0][i] = _mm256_mul_ps(vec[0][i], priors_w);
        vec[0][i] = _mm256_add_ps(vec[0][i], priors_x);

        vec[2][i] = vec[0][i];

        //float decoded_y = prior_y + loc_pred[1] * var_y * prior_h;
        vec[1][i] = _mm256_mul_ps(vec[1][i], vars_x_y);
        vec[1][i] = _mm256_mul_ps(vec[1][i], priors_h);
        vec[1][i] = _mm256_add_ps(vec[1][i], priors_y);

        vec[3][i] = vec[1][i];

        // decoded_ymin
        vec[0][i] = _mm256_sub_ps(vec[0][i], decoded_w);
        // decoded_ymin
        vec[1][i] = _mm256_sub_ps(vec[1][i], decoded_h);
        // decoded_xmax
        vec[2][i] = _mm256_add_ps(vec[2][i], decoded_w);
        // decoded_ymax
        vec[3][i] = _mm256_add_ps(vec[3][i], decoded_h);
    }
}

  
    