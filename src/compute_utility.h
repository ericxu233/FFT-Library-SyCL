#pragma once
#include "include/syclfft.h"
#include <cmath>
#define PI 3.1415926

/*
    note that this file is designed to do some fast mathematical operations

    - fast e power
*/

size_t bitReverse(size_t x, size_t length) {

    unsigned int rev = 0;
     
    // traversing bits of 'n' from the right
    int counter = length;
    while (counter > 0) {
        rev <<= 1;
        rev |= (x & 1);
        x >>= 1;
        counter--;
    }
    /* note: this function really needs testing */
    return rev;
}


inline void w_calculator(int base, int power, float& real, float& complex) {
    //calcuates the w notation value in fft
    if (power == 0) {
        real = 1;
        complex = 0;
        return;
    }

    real = sycl::cos(-2.0*PI*power/base);
    complex = sycl::sin(-2.0*PI*power/base);

}