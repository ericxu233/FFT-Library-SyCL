#include "include/syclfft.h"
#include <vector>

int main() {
    vector<float> data = {1, 2, 1, 3, 1, 2, 1, 2, 2, 2, 4, 1, 2, 2, 6, 0};
    vector<float> output_r;
    vector<float> output_c;


    //vector<float> data(1024*8, 0);
    
    //for (int i = 0; i < data.size(); i++) {
    //k    data[i] = i%2 + 1;
    //} 

    sycl_fft_setup();

    fft_max_max(data, output_r, output_c);
    //rs_parrallel(data, output_r, output_c);


    //cout << "( " << output_r[0] << " , " << output_r[data.size()/2] << " )" << endl;
    for (size_t i = 0; i < 16; i++) {
        cout << "( " << output_r[i] << " , " << output_c[i] << " )" << endl; 
    }
    //damn
    //damn another mileston

    return 0;
}