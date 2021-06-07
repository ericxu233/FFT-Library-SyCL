#include <iostream>
#include <CL/sycl.hpp>


namespace sycl = cl::sycl;

int main() {
    sycl::device device = sycl::default_selector{}.select_device();

    auto work_group_size = device.get_info<sycl::info::device::max_work_group_size>();
    std::cout << "This is max work group size: " << work_group_size << std::endl;

    auto max_work_item = device.get_info<sycl::info::device::max_work_group_size>();
    std::cout << "First dimension: " << max_work_item[1] << " Second dimension: " << max_work_item[2] << " Third dimesnion: " << max_work_item[3] << std::endl;
    //continue and test it on research cluster
}