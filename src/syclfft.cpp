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

class single_workgroup; 

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
    size_t tempp = length;

    while (tempp != 1) {
        tempp /= 2;
        stages++; 
    } 
    cout << stages << endl;

    cout << "work groups are " << Devicespec::work_group_size << ", items in work groups are " << Devicespec::dim1 << endl;



    sycl::device device = sycl::default_selector{}.select_device();
    
    //sycl::queue queue(sycl::default_selector{});

    
    sycl::queue queue(device, [] (sycl::exception_list el) {
       for (auto ex : el) { std::rethrow_exception(ex); }
    } );
    
    cout << "sycl exception setup is working" << endl;

    for (size_t i = 0; i < length; i++) {
        cout << bitReverse(i, stages) << endl;
    }

    {
        sycl::buffer<float, 1> buff_data(data.data(), sycl::range<1>(data.size()));
        sycl::buffer<float, 1> buff_real(temp_real.data(), sycl::range<1>(temp_real.size()));
        sycl::buffer<float, 1> buff_complex(temp_complex.data(), sycl::range<1>(temp_complex.size()));

        sycl::buffer<float, 1> buff_real_wr(real.data(), sycl::range<1>(real.size()));
        sycl::buffer<float, 1> buff_comp_wr(complex.data(), sycl::range<1>(complex.size()));

        /*
        {
            auto data_acc = buff_data.get_access<sycl::access::mode::read>();
            for (int i = 0; i < 16; i++) {
                cout << "( " << data_acc[i] << " )" << endl;
            }
        }
        */

        queue.submit([&] (sycl::handler& cgh) {
            auto data_acc = buff_data.get_access<sycl::access::mode::read>(cgh); //read only input data
            auto real_acc = buff_real.get_access<sycl::access::mode::read_write>(cgh);
            auto complex_acc = buff_complex.get_access<sycl::access::mode::read_write>(cgh);
            sycl::stream out(1024, 256, cgh);
            //now is the hard part, the parallel sycl algorithm
            cgh.parallel_for<class setup_kernal>(
                sycl::range<1>(length), [=] (sycl::id<1> i) {
                    size_t temp_index = bitReverse(i, stages);
                    out << temp_index << sycl::endl;
                    real_acc[i] = 0;
                    real_acc[i + length] = 0;
                    complex_acc[i] = 0;
                    complex_acc[i + length] = 0;
                    real_acc[i] = data_acc[temp_index];
                }
            );
        });
        queue.wait_and_throw();

        {
            auto real_acc = buff_real.get_access<sycl::access::mode::read_write>();
            auto complex_acc = buff_complex.get_access<sycl::access::mode::read_write>();
        
            for (int i = 0; i < length*2; i++) {
                cout << "( " << real_acc[i] << " , " << complex_acc[i] << " )" << endl;
            }
        }
        

        cout << "copy data is set up" << endl;
        
        for (size_t i = 1; i <= stages; i++) {
            cout << "omg" << endl;
            queue.submit([&] (sycl::handler& cgh) {
                auto real_acc = buff_real.get_access<sycl::access::mode::read_write>(cgh);
                auto complex_acc = buff_complex.get_access<sycl::access::mode::read_write>(cgh);

                //now is the hard part, the parallel sycl algorithm
                cgh.parallel_for<class fft_kernal>(
                    sycl::range<1>(length), [=] (sycl::id<1> j) {

                        int interval = 1;
                        interval <<= i;
                        
                        int offset_read = 0;
                        int offset_write = 0;

                        if (i%2 == 0) {
                            offset_read = 0;
                            offset_write = length;
                        }
                        else {
                            offset_read = length;
                            offset_write = 0;
                        }
                        
                        int tt_f = (j/(interval >> 1))%2;
                        if (tt_f == 0) {
                            float t_real = 0;
                            float t_complex = 0;
                            int power = (j%interval) * (length/interval);
                            w_calculator(length, power, t_real, t_complex);

                            complex_calculator(real_acc[j + offset_write + (interval >> 1)], complex_acc[j + offset_write + (interval >> 1)], t_real, t_complex);
                            real_acc[j + offset_read] = real_acc[j + offset_write] + t_real;
                            complex_acc[j + offset_read] = complex_acc[j + offset_write] + t_complex;
                        }
                        else {
                            float t_real = 0;
                            float t_complex = 0;
                            int power = (j%interval) * (length/interval);
                            w_calculator(length, power, t_real, t_complex);
                            complex_calculator(real_acc[j + offset_write], complex_acc[j + offset_write], t_real, t_complex);
                            real_acc[j + offset_read] = t_real + real_acc[j + offset_write - (interval >> 1)];
                            complex_acc[j + offset_read] = t_complex + complex_acc[j + offset_write - (interval >> 1)];
                        }
                        
                    }
                );
            });//needs to copy back to results!!!
            queue.wait_and_throw();
        }
        
        cout << "core parallelism is done" << endl;
        //damn
        queue.submit([&] (sycl::handler& cgh) {
            auto real_acc = buff_real.get_access<sycl::access::mode::read>(cgh);
            auto complex_acc = buff_complex.get_access<sycl::access::mode::read>(cgh);

            auto real_wr = buff_real_wr.get_access<sycl::access::mode::write>(cgh);
            auto comp_wr = buff_comp_wr.get_access<sycl::access::mode::write>(cgh);

            cgh.parallel_for<class finish_kernal>(
                sycl::range<1>(length), [=] (sycl::id<1> j) {
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

void fft_group_size(vector<float>& data, vector<float>& real, vector<float>& imag) {
    const size_t fft_length = data.size();
    
    size_t tempp = fft_length;

    size_t stages = 0;

    while (tempp != 1) {
        tempp /= 2;
        stages++; 
    }

    real.resize(data.size());
    complex.resize(data.size());

    sycl::device device = sycl::default_selector{}.select_device();
    
    //sycl::queue queue(sycl::default_selector{});

    
    sycl::queue queue(device, [] (sycl::exception_list el) {
       for (auto ex : el) { std::rethrow_exception(ex); }
    });



    {
        sycl::buffer<float, 1> buff_data(data.data(), sycl::range<1>(data.size()));
        sycl::buffer<float, 1> buff_real(real.data(), sycl::range<1>(data.size()));
        sycl::buffer<float, 1> buff_imag(image.data(), sycl::range<1>(data.size()));

        queue.submit([&] (sycl::handler& cgh){
            sycl::accessor <float, 1, sycl::access::mode::read_write, sycl::access::target::local>
                         local_real(sycl::range<1>(fft_length), cgh);
            sycl::accessor <float, 1, sycl::access::mode::read_write, sycl::access::target::local>
                         local_imag(sycl::range<1>(fft_length), cgh);
            


            auto read_data = buff_data.get_access<sycl::access::mode::read>(cgh);
            auto real_acc = buff_real.get_access<sycl::access::mode::write>(cgh);
            auto imag_acc = buff_imag.get_access<sycl::access::mode::write>(cgh);

            cgh.parallel_for<class single_workgroup>(
                sycl::nd_range<1> (fft_length, fft_length),
                [=] (sycl::nd_item<1> item) {

                    size_t index = item.get_local_linear_id();
                    size_t reverse_index = bitReverse(index);
                    local_real[index] = 0;
                    local_imag[index] = 0;
                    local_real[index] = read_data[reverse_index];

                    //synchronize
                    item.barrier(sycl::access::fence_space::local_space);
                    //...

                    for (int i = 1; i <= stages; i++) {
                        int interval = 1;
                        interval <<= i;

                        int tt_f = (index/(interval >> 1))%2;

                        if (tt_f == 0) {
                            float t_real = 0;
                            float t_complex = 0;
                            int power = (index%interval) * (fft_length/interval);
                            w_calculator(fft_length, power, t_real, t_complex);
                            complex_calculator(local_real[index + (interval >> 1)], local_imag[index + (interval >> 1], t_real, t_complex);

                            //synchronize
                            item.barrier(sycl::access::fence_space::local_space);
                            //...

                            local_real[index] = local_real[index] + t_real;
                            local_imag[index] = local_imag[index] + t_complex;

                            //synchronize
                            item.barrier(sycl::access::fence_space::local_space);
                            //...
                        }
                        else {
                            float t_real = 0;
                            float t_complex = 0;
                            int power = (index%interval) * (fft_length/interval);
                            w_calculator(fft_length, power, t_real, t_complex);
                            float fence_add_r = local_real[index - (interval >> 1)];
                            float fence_add_i = local_imag[index - (interval >> 1)];

                            //synchronize
                            item.barrier(sycl::access::fence_space::local_space);
                            //...

                            local_real[index] = t_real + fence_add_r;
                            local_imag[index] = t_imag + fence_add_i;

                            //synchronize
                            item.barrier(sycl::access::fence_space::local_space);
                            //...
                        }
                    }

                    real_acc[index] = local_real[index];
                    imag_acc[index] = local_imag[index];

                }
            );
        });

    }


}
