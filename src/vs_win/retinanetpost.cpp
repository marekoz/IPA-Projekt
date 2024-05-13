
#include <cstring>
#include <string>
#include <iostream>
#include <inttypes.h>
#include "ipa_tool.h"

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;



#include "prior_boxes.hpp"
#include "utils.hpp"
#include "reader.hpp"



#include <iterator>
#include <fstream>



//example: call extern function
extern "C" { void f1(int a);}

#define CONFIDENCE_THRESHOLD 0.999
#define INPUT_WIDTH 640
#define INPUT_HEIGHT 480
#define ANCHORS_COUNT 12600

int main(int argc, char** argv)
{
	if (argc != 3)
	{
		std::cout << "Run program by: ./retinapost input/vector.txt input/image.png";
	}

	Mat image = imread(argv[1]);

	if (image.empty()) 
	{
        cout << "Could not open or find a image" << endl;
        return -1;
    }

	//All constants refer to the configuration used in prior_boxes.cpp and to the 640x480 resolution.
	std::vector<int> image_size = {INPUT_WIDTH, INPUT_HEIGHT};
	std::vector<float> variances = {0.1f, 0.2f};
	size_t total0_len = ANCHORS_COUNT*4;
	size_t total1_len = ANCHORS_COUNT*2;
	size_t num_anchors = total0_len / 4;

	PriorBox priorBox(image_size, "projekt");

    std::vector<std::vector<float>>  priors = priorBox.forward();

    Scalar color(0, 255, 0); // Color of the rectangle (in BGR)
    int thickness = 2; // Thickness of the rectangle border

	InstructionCounter counter;
	
	/*******************Part to optmize*********************/
    const int ROWS_COUNT = 12600;
    const int AVX_ROWS_COUNT = 1575; //12600/8 ,avx 8 bytes each __m256
    std::cout << "READ: ";
    counter.start();
    __m256 **vec = (__m256 **)_mm_malloc(5 * sizeof(__m256 *), 32); // 32-byte alignment for __m256;
    for (int i = 0; i < 5; ++i) {
        vec[i] = (__m256 *)_mm_malloc(AVX_ROWS_COUNT * sizeof(__m256), 32);
    }

	readFloatsFromFile(argv[2], vec); 
    // this function creates struct of arrays from arrays of structs (its 2d array)
    // the array after the loop looks like this 
    
    // vec[0][[x1,...,x8],[x9,...,x16],...,[...x12600]]
    // vec[1][[y1,...,y8],[y9,...,y16],...,[...y12600]]
    // vec[2][[w1,...,w8],[w9,...,w16],...,[...w12600]]
    // vec[3][[h1,...,h8],[h9,...,h16],...,[...h12600]]
    // vec[4][[score1,...,score8],[score9,...,score16],...,[...score12600]]
    // it prepares for decode() function that uses avx intrinsic instructions
    counter.print(); 



    cout << "DECODE: ";
    counter.start();
    decode(vec, priors, variances); // decoded boxes == vec
    counter.print();

    // vec[0][[decoded_xmin1,...]...]
    // vec[1][[decoded_ymin1,...]...]
    // vec[2][[decoded_xmax1,...]...]
    // vec[3][[decoded_ymax1,...]...]


    std::cout << "FILTER BY SCORE + SCALE RESOLUTION: ";
    counter.start();

    __m256 *det_boxes;
    det_boxes = (__m256*)_mm_malloc(ROWS_COUNT * sizeof(__m256), 32); // 32-byte alignment for __m256
    size_t det_boxes_cnt = 0;
    __m256 conf_tresh = _mm256_set1_ps(CONFIDENCE_THRESHOLD);
    __m256 resolution = _mm256_set_ps(0.0f, 0.0f, 0.0f, 1.0f, 480.0f,  640.0f, 480.0f, 640.0f);
    for (size_t i = 0; i < AVX_ROWS_COUNT; i++)
    {
        __m256 result = _mm256_cmp_ps(vec[4][i], conf_tresh, _CMP_GT_OQ);
        for (size_t j = 0; j < 8; j++)
        {   
            if (result[j]) {
                det_boxes[det_boxes_cnt] = _mm256_set_ps(0.0f, 0.0f, 0.0f, vec[4][i][j], vec[3][i][j], vec[2][i][j], vec[1][i][j], vec[0][i][j]);
                det_boxes[det_boxes_cnt] = _mm256_mul_ps(det_boxes[det_boxes_cnt], resolution);
                det_boxes_cnt++;
            }
        }
    }
    counter.print();
    


    std::cout << "NMS: ";
    counter.start();
    det_boxes_cnt = nms(det_boxes, det_boxes_cnt, 0.4);
    counter.print();


    // //Test
	// //f1(10);

	// counter.print();

	// /************************************************/


    for (int i = 0; i < det_boxes_cnt; i++)
    {
		#ifdef DEBUG
			printf("Box %f %f %f %f %f\n", det_boxes[i][0],  det_boxes[i][1],  det_boxes[i][2],  det_boxes[i][3],  det_boxes[i][4]);
		#endif


        cv::Rect roi((int)det_boxes[i][0],  (int)det_boxes[i][1],  (int)det_boxes[i][2]- (int)det_boxes[i][0], (int)det_boxes[i][3] - (int)det_boxes[i][1]);
		rectangle(image, roi, color, thickness);
    }

    imshow("Output", image);
    waitKey(0);


    for (int i = 0; i < 5; ++i) {
    _mm_free(vec[i]);
    }
    _mm_free(vec);
    _mm_free(det_boxes);
	return 0;
}