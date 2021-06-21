#include "include/syclfft.h"
#include <vector>

int main() {
    vector<float> data = {1 , 2, 1, 3, 1, 2, 1, 2};
    vector<float> output_r;
    vector<float> output_c;

    sycl_fft_setup();

    fft_group_size(data, output_r, output_c);

    for (size_t i = 0; i < output_c.size(); i++) {
        cout << "( " << output_r[i] << " , " << output_c[i] << " )" << endl; 
    }
    //damn


    return 0;
}