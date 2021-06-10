
#include "include/syclfft.h"


namespace Devicespec {
    size_t work_group_size;

    //each work group maximum items in each dimension
    size_t dim1;
    size_t dim2;
    size_t dim3;

    size_t total_tims_dim1;

    bool cpu_device;
}


void sycl_fft_setup() {
    sycl::device device = sycl::default_selector{}.select_device();

    Devicespec::work_group_size = device.get_info<sycl::info::device::max_work_group_size>();

    auto max_work_item = device.get_info<sycl::info::device::max_work_item_sizes>();
    Devicespec::dim3 = max_work_item[0];
    Devicespec::dim2 = max_work_item[1];
    Devicespec::dim1 = max_work_item[2];


    if (Devicespec::work_group_size == 1) cpu_device = true;
    else cpu_device = false;
}

void rs_parrallel(vector<float>& data, vecotr<float>& real, veector<float>& complex) {
    
}