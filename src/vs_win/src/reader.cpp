/*
* Function to read floats from txt
* Tomas Goldmann,2023
*/


#include "reader.hpp"


#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iterator>

using namespace std;



const int numerals = 8;
void readFloatsFromFile(const string& filename,__m256 **vec) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    //read x,y,w,h
    char buffer[32];
    float floats[4][8];
    
    int l = 0;
    int m = 0;

    size_t p = 0;
    char c;
    bool skip = false;
    size_t i = 0;
    size_t j = 0;
    while (p < 1575) {
        file.get(c);
        if (c == ' ')
        {
            continue;
        }
        if (c == 10 || c == ',')
        {
            skip = false;
            buffer[i] = '\0';
            floats[l][m] = stof(buffer);
            l = (l + 1) % 4;
            j++;
            if (j % 4 == 0)
            {
                m++;
            }
            if (j == 32)
            {
                vec[0][p]= _mm256_loadu_ps(floats[0]);
                vec[1][p]= _mm256_loadu_ps(floats[1]);
                vec[2][p]= _mm256_loadu_ps(floats[2]);
                vec[3][p]= _mm256_loadu_ps(floats[3]);
                p++;
                m = 0;
                j = 0;
            }
            i = 0;
            buffer[0] = '\0';
        }
        else if (!skip)
        {
            buffer[i] = c;
            i++;
            if (i >= numerals)
            {
                skip = true;
            }
        }
    }


    // read score 
    float score_buff[8];
    skip = false;
    bool odd = false;
    p = 0;
    i = 0;
    j = 0;
    while(file.get(c))
    {
        if (c == ' ')
        {
            continue;
        }
        if (odd)
        {
            if (c == 10 || c == ',')
            {
                buffer[i] = '\0';
                score_buff[j] = stof(buffer);
            
                j++;
                if (j == 8)
                {
                    vec[4][p]= _mm256_loadu_ps(score_buff);
                    p++;
                    j = 0;

                }
                skip = false;
                odd = false;
                i = 0;
                buffer[0] = '\0';
            }
            else if (!skip)
            {
                buffer[i] = c;
                i++;
                if (i >= numerals)
                {
                    skip = true;
                }
            }
        }
        else
        {
            if (c == 10 || c == ',')
            {
                odd = true;
            }
        }
    }
    buffer[i] = '\0';
    score_buff[j] = stof(buffer);
    vec[4][p]= _mm256_loadu_ps(score_buff);
}
