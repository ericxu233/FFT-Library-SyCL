#pragma once
#include <vector>
#include <CL/sycl.hpp>

namespace sycl = cl::sycl;
using namespace std;



void sycl_fft_setup();

//rs means result split, essentailly meaning that the output will be split to real and imaginary parts
void rs_parrallel(vector<float>& data, vecotr<float>& real, veector<float>& complex); 
void rs_sequencial(vector<float>& data, vecotr<float>& real, veector<float>& complex);