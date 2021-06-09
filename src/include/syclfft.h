#pragma once
#include <vector>
#include <CL/sycl.hpp>

namespace sycl = cl::sycl;
using namespace std;



void sycl_fft_setup();
void c_parrallel(vector<float>& data);
void c_sequencial(vector<float>& data);