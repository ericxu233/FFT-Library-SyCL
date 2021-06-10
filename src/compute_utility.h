#pragma once

#include <cmath>

/*
    note that this file is designed to do some fast mathematical operations

    - fast e power
*/

inline unsigned int bitReverse(unsigned int x, int loglength) {

    unsigned int rev = 0;
     
    // traversing bits of 'n' from the right
    while (x > 0) {
        rev <<= 1;
        rev |= (x & 1);
        x >>= 1;
    }
    /* note: this function really needs testing */
    return rev;
}
