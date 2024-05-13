#include "utils.hpp"


bool CompareBBox(__m256 & a, const __m256 & b)
{
    return a[4] > b[4];
}

size_t nms(__m256* (&bboxes), int32_t size, float threshold) {
    size_t bboxes_nms_cnt = 0;
    std::vector<int32_t> mask_merged(size, 0);
    std::sort(&bboxes[0], &bboxes[size], CompareBBox);

    for (int32_t i = 0; i < size; i++) {
        if (mask_merged[i] == 1)
            continue;

        bboxes[bboxes_nms_cnt] = bboxes[i];
        bboxes_nms_cnt++;

        float x1 = (bboxes[i][0]);
        float y1 = (bboxes[i][1]);
        float x2 = (bboxes[i][2]);
        float y2 = (bboxes[i][3]);
        float area1 = (x2 - x1 + 1) * (y2 - y1 + 1);

        for (int32_t j = i + 1; j < size; j++) {
            if (mask_merged[j] == 1)
                continue;

            float x = std::max(x1, (bboxes[j][0]));
            float y = std::max(y1, (bboxes[j][1]));
            float w = std::min(x2, (bboxes[j][2])) - x + 1;
            float h = std::min(y2, (bboxes[j][3])) - y + 1;

            if (w <= 0 || h <= 0)
                continue;

            float area2 = (bboxes[j][2] - bboxes[j][0] + 1) * (bboxes[j][3] - bboxes[j][1] + 1);
            float area_intersect = w * h;

            if (area_intersect / (area1 + area2 - area_intersect) > threshold)
                mask_merged[j] = 1;
        }
    }

    return bboxes_nms_cnt;
}
