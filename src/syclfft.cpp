#include "include/syclfft.h"
#include "compute_utility.h"


namespace Devicespec {
    size_t work_group_size;

    //each work group maximum items in each dimension
    size_t dim1;
    size_t dim2;
    size_t dim3;

    size_t total_tims_dim1;

    bool cpu_device;
}

class fft_kernal;
class setup_kernal;
class finish_kernal;

void sycl_fft_setup() {
    sycl::device device = sycl::default_selector{}.select_device();

    Devicespec::work_group_size = device.get_info<sycl::info::device::max_work_group_size>();

    auto max_work_item = device.get_info<sycl::info::device::max_work_item_sizes>();
    Devicespec::dim3 = max_work_item[0];
    Devicespec::dim2 = max_work_item[1];
    Devicespec::dim1 = max_work_item[2];


    if (Devicespec::work_group_size == 1) Devicespec::cpu_device = true;
    else Devicespec::cpu_device = false;
}

void rs_parrallel(vector<float>& data, vector<float>& real, vector<float>& complex) {
    //this stage of development assumes that data does not exceed max number of work items
    real.resize(data.size());
    complex.resize(data.size());
    
    vector<float> temp_real(2*data.size());
    vector<float> temp_complex(2*data.size());

    size_t stages = 0;
    size_t length = data.size();
    size_t length2 = 2*length;

    while (length != 1) {
        length /= 2;
        stages++; 
    } 
    cout << stages << endl;

    cout << "work groups are " << Devicespec::work_group_size << ", items in work groups are " << Devicespec::dim1 << endl;


    sycl::device device = sycl::default_selector{}.select_device();

    sycl::queue queue(device, [] (sycl::exception_list el) {
       for (auto ex : el) { std::rethrow_exception(ex); }
    } );
    
    cout << "sycl exception setup is working" << endl;

    {
        sycl::buffer<float, 1> buff_data(data.data(), sycl::range<1>(data.size()));
        sycl::buffer<float, 1> buff_real(temp_real.data(), sycl::range<1>(temp_real.size()));
        sycl::buffer<float, 1> buff_complex(temp_complex.data(), sycl::range<1>(temp_complex.size()));

        sycl::buffer<float, 1> buff_real_wr(real.data(), sycl::range<1>(real.size()));
        sycl::buffer<float, 1> buff_comp_wr(complex.data(), sycl::range<1>(complex.size()));

        queue.submit([&] (sycl::handler& cgh) {
            auto data_acc = buff_data.get_access<sycl::access::mode::read>(cgh); //read only input data
            auto real_acc = buff_real.get_access<sycl::access::mode::read_write>(cgh);
            auto complex_acc = buff_complex.get_access<sycl::access::mode::read_write>(cgh);

            //now is the hard part, the parallel sycl algorithm
            cgh.parallel_for<class setup_kernal>(
                sycl::range<1>(length2), [=] (sycl::id<1> i) {
                    int temp_index = bitReverse(i);
                    real_acc[i] = 0;
                    complex_acc[i] = 0;
                    if (i < length) real_acc[i] = data_acc[temp_index];
                }
            );
        });
        queue.wait_and_throw();

        cout << "copy data is set up" << endl;
        /*
        for (int i = 1; i < stages; i++) {
            queue.submit([&] (sycl::handler& cgh) {
                auto real_acc = buff_real.get_access<sycl::access::mode::read_write>(cgh);
                auto complex_acc = buff_complex.get_access<sycl::access::mode::read_write>(cgh);

                //now is the hard part, the parallel sycl algorithm
                cgh.parallel_for<class fft_kernal>(
                    sycl::range<1>(length), [=] (sycl::id<1> j1) {

                        int interval = 2;
                        interval <<= i;

                        int offset_read = i%2 ? 0 : length;
                        int offset_write = i%2 ? length : 0; 

                        if ((j/(interval >> 1))%2 == 0) {
                            float t_real = 0;
                            float t_complex = 0;
                            int power = (j%interval) * (length/interval);
                            w_calculator(length, power, t_real, t_complex);
                            real_acc[j + offset_read] = real_acc[j + offset_write] + t_real*real_acc[j + offset_write + (interval >> 1)];
                            complex_acc[j + offset_read] = complex_acc[j + offset_write] + t_complex*complex_acc[j + offset_write + (interval >> 1)];
                        }
                        else {
                            float t_real = 0;
                            float t_complex = 0;
                            int power = (j%interval) * (length/interval);
                            w_calculator(length, power, t_real, t_complex);
                            real_acc[j + offset_read] = t_real*real_acc[j + offset_write] + real_acc[j + offset_write - (interval >> 1)];
                            complex_acc[j + offset_read] = t_complex*complex_acc[j + offset_write] + complex_acc[j + offset_write - (interval >> 1)];
                        }
                    }
                );
            });//needs to copy back to results!!!
            queue.wait_and_throw();
        }
        */
        queue.submit([&] (sycl::handler& cgh) {
            auto real_acc = buff_real.get_access<sycl::access::mode::read>(cgh);
            auto complex_acc = buff_complex.get_access<sycl::access::mode::read>(cgh);

            auto real_wr = buff_real_wr.get_access<sycl::access::mode::write>(cgh);
            auto comp_wr = buff_comp_wr.get_access<sycl::access::mode::write>(cgh);

            cgh.parallel_for<class finish_kernal>(
                sycl::range<1>(length2), [=] (sycl::id<1> j) {
                    if (stages%2 == 0) {
                        real_wr[j] = real_acc[j];
                        comp_wr[j] = complex_acc[j];
                    }
                    else {
                        real_wr[j] = real_acc[j + length];
                        comp_wr[j] = complex_acc[j + length];
                    }
                }
            );
        });
    }
}
