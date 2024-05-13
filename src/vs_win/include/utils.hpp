/*
* Non-maximum suppression
* Tomas Goldmann,2024
*/



#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "immintrin.h"

size_t nms(__m256* (&bboxes), int32_t size, float threshold);
