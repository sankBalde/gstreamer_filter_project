#include "filter_impl.h"

#include <cassert>
#include <chrono>
#include <thread>
#include <cstdio>
#include <cstdint>
#include <vector>
#include "cudautils.cuh"

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "CUDA Runtime Error at: %s: %d\n", file, line);
        std::fprintf(stderr, "%s %s\n", cudaGetErrorString(err), func);
        // We don't exit when we encounter CUDA errors in this example.
        std::exit(EXIT_FAILURE);
    }
}


uint8_t* getPointerToRGB(RGB* rgb) {
    return reinterpret_cast<uint8_t*>(rgb);
}

RGB* getRGBFromPointer(uint8_t* pointer) {
    return reinterpret_cast<RGB*>(pointer);
}


std::vector<RGB> convertToVector(RGB* array, size_t size) {
    std::vector<RGB> vec;
    for (size_t i = 0; i < size; i++) {
        vec.push_back(array[i]);
    }
    return vec;
}






int ff = 1;
static std::vector<uint8_t> global_buffer;
static std::vector<RGBImage> background_images;
static int bg_number_frame = 10;
static int cpt_frame = 0;

extern "C" {
    void filter_impl(uint8_t* src_buffer, int width, int height, int src_stride, int pixel_stride)
    {
        cpt_frame++;
        size_t buffer_size = height * src_stride;

        if (global_buffer.empty())
        {
            global_buffer.assign(src_buffer, src_buffer + buffer_size);
            return;
        }

     
        RGBImage h_rgbImage1(width, height);
        RGBImage h_rgbImage2(width, height);

        std::vector<RGB> prev_rgb_image_vect = uint8_to_rgb(global_buffer.data(), width, height);
        std::vector<RGB> new_rgb_image_vect = uint8_to_rgb(src_buffer, width, height);

        h_rgbImage1.buffer = std::move(prev_rgb_image_vect);
        h_rgbImage2.buffer = std::move(new_rgb_image_vect);

        background_images.push_back(h_rgbImage2);
        if (cpt_frame == bg_number_frame-1)
        {
            cpt_frame = 0;
            background_images.push_back(h_rgbImage1);
            RGBImage average = averageRGBImages(background_images);
            background_images.clear();
            uint8_t* average_buffer_ptr = rgb_to_uint8(average.buffer);
            global_buffer.assign(average_buffer_ptr, average_buffer_ptr + buffer_size);
        }


        RGB *d_rgbImage1, *d_rgbImage2;
        Lab *d_labImage1, *d_labImage2;


        cudaMalloc(&d_rgbImage1, width * height * sizeof(RGB));
        cudaMalloc(&d_rgbImage2, width * height * sizeof(RGB));
        cudaMalloc(&d_labImage1, width * height * sizeof(Lab));
        cudaMalloc(&d_labImage2, width * height * sizeof(Lab));



        cudaMemcpy(d_rgbImage1, h_rgbImage1.buffer.data(), width * height * sizeof(RGB), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rgbImage2, h_rgbImage2.buffer.data(), width * height * sizeof(RGB), cudaMemcpyHostToDevice);

        int blockSize = 256;
        int numBlocks = (width * height + blockSize - 1) / blockSize;

        RGBtoLabKernel<<<numBlocks, blockSize>>>(d_rgbImage1, d_labImage1, width * height);
        RGBtoLabKernel<<<numBlocks, blockSize>>>(d_rgbImage2, d_labImage2, width * height);

        cudaDeviceSynchronize();

        Mask distance_lab(width, height);
        computeDeltaE(d_labImage1, d_labImage2, distance_lab, width, height);


        double *d_distance_lab, *d_opened_lab;
        cudaMalloc(&d_distance_lab, width * height * sizeof(double));
        cudaMalloc(&d_opened_lab, width * height * sizeof(double));

        
        cudaMemcpy(d_distance_lab, distance_lab.buffer.data(), width * height * sizeof(double), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        int morph_radius = 3;  
        morphological_opening_cuda(d_distance_lab, d_opened_lab, width, height, morph_radius);
        cudaDeviceSynchronize();


        cudaMemcpy(distance_lab.buffer.data(), d_opened_lab, width * height * sizeof(double), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();



        Mask opened_mask = apply_hysteresis_threshold_cuda(distance_lab, 4, 30);
        cudaDeviceSynchronize();

        RGBImage h_output_rgb = apply_mask_to_rgb(opened_mask, h_rgbImage2);
        cudaDeviceSynchronize();


        uint8_t *buf = rgb_to_uint8(h_output_rgb.buffer);


        for (size_t k = 0; k < height *src_stride;k++)
        {
            src_buffer[k] = buf[k];
           
            
        }

        cudaFree(d_rgbImage1);
        cudaFree(d_rgbImage2);
        cudaFree(d_labImage1);
        cudaFree(d_labImage2);
        cudaFree(d_distance_lab);
        cudaFree(d_opened_lab);

    }
}






