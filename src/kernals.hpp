#include "include/syclfft.h"
#include "compute_utility.h"

//for length smaller than group item size
class group_reduction {
    using read_acc =
		    sycl::accessor<float, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;

    using write_acc =
		    sycl::accessor<float, 1, sycl::access::mode::write, sycl::access::target::global_buffer>;

    using rrww_acc =
		    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;

    using rewr_acc =
		    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local>;



public:
    //note stages should be 10 in this case!!!
    group_reduction(size_t length, size_t stagesR, read_acc dataR, rrww_acc realR, 
    rrww_acc imagR, rewr_acc local_realR, rewr_acc local_imagR, size_t offset, size_t phase2_stages, sycl::stream out):
    fft_length(length), stages(stagesR), data(dataR), real(realR), imag(imagR),
    local_real(local_realR), local_imag(local_imagR), offset(offset), phase2_stages(phase2_stages), out(out)
    {}

    void operator()(sycl::nd_item<1> item) const{
        size_t index1 = item.get_local_linear_id();
        //index1 means local index
        size_t sub_index = item.get_global_linear_id(); //sub index is for this
        size_t global_index = item.get_global_linear_id() + offset;
        size_t reverse_index = bitReverse(global_index, stages);
        size_t local_range = item.get_local_range();
        size_t group_range = item.get_group_range();
        local_real[index1] = 0;
        local_imag[index1] = 0;
        local_real[index1] = data[reverse_index];
                    
        //synchronize
        item.barrier(sycl::access::fence_space::local_space);
        //...

        for (size_t i = 1; i <= stages; i++) {
            int interval = 1;
            interval <<= i;

            int tt_f = (index1/(interval >> 1))%2;

                        
            float t_real = 0;
            float t_complex = 0;

            float fence_add_r = 0;
            float fence_add_i = 0;

            int power = (index1%interval) * (fft_length/interval);
            w_calculator(fft_length, power, t_real, t_complex);
                        
            if (tt_f == 0) {
                complex_calculator(local_real[index1 + (interval >> 1)], local_imag[index1 + (interval >> 1)], t_real, t_complex);
            }
            else {
                complex_calculator(local_real[index1], local_imag[index1], t_real, t_complex);
                fence_add_r = local_real[index1 - (interval >> 1)];
                fence_add_i = local_imag[index1 - (interval >> 1)];
            }
                        
            //synchronize
            item.barrier(sycl::access::fence_space::local_space);
            //...

            if (tt_f == 0) {
                local_real[index1] = local_real[index1] + t_real;
                local_imag[index1] = local_imag[index1] + t_complex;
            }
            else {
                local_real[index1] = t_real + fence_add_r;
                local_imag[index1] = t_complex + fence_add_i;
            }

            //synchronize
            item.barrier(sycl::access::fence_space::local_space);
            //...

        }

        real[global_index] = local_real[index1];
        imag[global_index] = local_imag[index1];
    }


private:
    size_t fft_length;
    size_t stages;
    read_acc data;
    rrww_acc real;
    rrww_acc imag;
    rewr_acc local_real;
    rewr_acc local_imag;
    size_t offset;
    size_t phase2_stages;
    sycl::stream out;
};

class phase2_reduction {
    using read_acc =
		    sycl::accessor<float, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;

    using write_acc =
		    sycl::accessor<float, 1, sycl::access::mode::write, sycl::access::target::global_buffer>;

    using rrww_acc =
		    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;

    using rewr_acc =
		    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local>;



public:
    //note stages should be 10 in this case!!!
    phase2_reduction(size_t length, size_t stagesR, read_acc dataR, rrww_acc realR, 
    rrww_acc imagR, rewr_acc local_realR, rewr_acc local_imagR, size_t offset, size_t phase2_stages, sycl::stream out):
    fft_length(length), stages(stagesR), data(dataR), real(realR), imag(imagR),
    local_real(local_realR), local_imag(local_imagR), offset(offset), phase2_stages(phase2_stages), out(out)
    {}

    void operator()(sycl::nd_item<1> item) const{
        size_t index1 = item.get_local_linear_id();
        //index1 means local index
        size_t sub_index = item.get_global_linear_id(); //sub index is for this
        size_t global_index = item.get_global_linear_id() + offset;
        size_t reverse_index = bitReverse(global_index, stages);
        size_t local_range = item.get_local_range(0);
        size_t group_range = item.get_group_range(0);
        local_real[index1] = 0;
        local_imag[index1] = 0;

        local_real[index1] = real[(index1%group_range)*local_range + index1/group_range + offset];
        local_imag[index1] = imag[(index1%group_range)*local_range + index1/group_range + offset];

        //synchronize
        item.barrier(sycl::access::fence_space::local_space);
        //...

        for (size_t i = 1; i <= phase2_stages; i++) {
            int interval = 1;
            interval <<= i;

            int tt_f = (index1/(interval >> 1))%2;

                        
            float t_real = 0;
            float t_complex = 0;

            float fence_add_r = 0;
            float fence_add_i = 0;

            int power = (index1%interval) * (fft_length/interval);
            w_calculator(fft_length, power, t_real, t_complex);
                        
            if (tt_f == 0) {
                complex_calculator(local_real[index1 + (interval >> 1)], local_imag[index1 + (interval >> 1)], t_real, t_complex);
            }
            else {
                complex_calculator(local_real[index1], local_imag[index1], t_real, t_complex);
                fence_add_r = local_real[index1 - (interval >> 1)];
                fence_add_i = local_imag[index1 - (interval >> 1)];
            }
                        
            //synchronize
            item.barrier(sycl::access::fence_space::local_space);
            //...

            if (tt_f == 0) {
                local_real[index1] = local_real[index1] + t_real;
                local_imag[index1] = local_imag[index1] + t_complex;
            }
            else {
                local_real[index1] = t_real + fence_add_r;
                local_imag[index1] = t_complex + fence_add_i;
            }

            //synchronize
            item.barrier(sycl::access::fence_space::local_space);
            //...
        }

        real[(index1%group_range)*local_range + index1/group_range + offset] = local_real[index1];
        imag[(index1%group_range)*local_range + index1/group_range + offset] = local_imag[index1];
    }


private:
    size_t fft_length;
    size_t stages;
    read_acc data;
    rrww_acc real;
    rrww_acc imag;
    rewr_acc local_real;
    rewr_acc local_imag;
    size_t offset;
    size_t phase2_stages;
    sycl::stream out;
};

class second_reduction{
    using rwg_acc =
		    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;


    using rewr_acc =
		    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local>;



public:
    second_reduction(size_t length, size_t stagesR, rwg_acc realR, 
    rwg_acc imagR, rewr_acc local_realR, rewr_acc local_imagR, size_t group, size_t offset, sycl::stream out):
    fft_length(length), stages(stagesR), real(realR), imag(imagR),
    local_real(local_realR), local_imag(local_imagR), stride(group), offset(offset), out(out)
    {}

    void operator()(sycl::nd_item<1> item) const{
        size_t group_id = item.get_group_linear_id();
        size_t global_index = item.get_global_linear_id();
        size_t local_index = item.get_local_linear_id();

        local_real[local_index] = real[local_index*stride + group_id];
        local_imag[local_index] = imag[local_index*stride + group_id];

        //synchronize
        item.barrier(sycl::access::fence_space::local_space);
        //...

        for (size_t i = 1; i <= stages; i++) {
            int interval = 1;
            interval <<= i;

            int tt_f = (local_index/(interval >> 1))%2;

                        
            float t_real = 0;
            float t_complex = 0;

            float fence_add_r = 0;
            float fence_add_i = 0;

            int power = (local_index%interval) * (fft_length/interval);
            w_calculator(fft_length, power, t_real, t_complex);
                        
            if (tt_f == 0) {
                complex_calculator(local_real[local_index + (interval >> 1)], local_imag[local_index + (interval >> 1)], t_real, t_complex);
            }
            else {
                complex_calculator(local_real[local_index], local_imag[local_index], t_real, t_complex);
                fence_add_r = local_real[local_index - (interval >> 1)];
                fence_add_i = local_imag[local_index - (interval >> 1)];
            }
                        
            //synchronize
            item.barrier(sycl::access::fence_space::local_space);
            //...

            if (tt_f == 0) {
                local_real[local_index] = local_real[local_index] + t_real;
                local_imag[local_index] = local_imag[local_index] + t_complex;
            }
            else {
                local_real[local_index] = t_real + fence_add_r;
                local_imag[local_index] = t_complex + fence_add_i;
            }

            //synchronize
            item.barrier(sycl::access::fence_space::local_space);
            //...
        }

        real[local_index*stride + group_id] = local_real[local_index];
        imag[local_index*stride + group_id] = local_imag[local_index];

    }

private:
    size_t fft_length;
    size_t stages;
    size_t stride;
    rwg_acc real;
    rwg_acc imag;
    rewr_acc local_real;
    rewr_acc local_imag;
    size_t offset;
    sycl::stream out;
};