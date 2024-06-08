#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include "image.hh" 
#include "utils.hh"




__device__ Lab RGBtoLab_device(RGB rgb) {
    float L = (0.2126 * rgb.R + 0.7152 * rgb.G + 0.0722 * rgb.B) / 255.0;
    float a = (0.5 * rgb.R - 0.419 * rgb.G - 0.081 * rgb.B) / 255.0;
    float b = (-0.169 * rgb.R - 0.331 * rgb.G + 0.5 * rgb.B) / 255.0;
    return {L * 100, a * 128 + 128, b * 128 + 128};
}

__device__ RGB LabtoRGB_device(Lab lab) {
    float y = (lab.L + 16) / 116.0;
    float x = lab.a / 500.0 + y;
    float z = y - lab.b / 200.0;
    float R = x * 3.2406;
    float G = y * -1.5372;
    float B = z * -0.4986;
    return {static_cast<unsigned char>(R * 255), static_cast<unsigned char>(G * 255), static_cast<unsigned char>(B * 255)};
}

        
        
__global__ void RGBtoLabKernel(RGB* d_rgb, Lab* d_lab, int numPixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPixels) {
        RGB rgb = d_rgb[idx];
        d_lab[idx] = RGBtoLab_device(rgb);
    }
}

__global__ void LabtoRGBKernel(Lab* d_lab, RGB* d_rgb, int numPixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPixels) {
        Lab lab = d_lab[idx];
        d_rgb[idx] = LabtoRGB_device(lab);
    }
}


__global__ void deltaE_cie76_kernel(const Lab* lab1, const Lab* lab2, double* output, int numPixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPixels) {
        Lab l1 = lab1[idx];
        Lab l2 = lab2[idx];
        double deltaL = l1.L - l2.L;
        double deltaA = l1.a - l2.a;
        double deltaB = l1.b - l2.b;
        output[idx] = sqrt(deltaL * deltaL + deltaA * deltaA + deltaB * deltaB);
    }
}

void computeDeltaE(Lab* d_labImage1, Lab* d_labImage2, Mask& mask, int width, int height) {
    int numPixels = width * height;
    double* d_output;
    cudaMalloc(&d_output, numPixels * sizeof(double));

    int blockSize = 256;
    int numBlocks = (numPixels + blockSize - 1) / blockSize;
    deltaE_cie76_kernel<<<numBlocks, blockSize>>>(d_labImage1, d_labImage2, d_output, numPixels);
    cudaDeviceSynchronize();

    cudaMemcpy(mask.buffer.data(), d_output, numPixels * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_output);
}


__global__ void dilate_kernel(double* input, double* output, int width, int height, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    double max_distance = -1e30; 
    for (int j = -radius; j <= radius; ++j) {
        for (int i = -radius; i <= radius; ++i) {
            int nx = x + i;
            int ny = y + j;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                double neighbor_distance = input[ny * width + nx];
                max_distance = max(max_distance, neighbor_distance);
            }
        }
    }

    output[y * width + x] = max_distance;
}

__global__ void erode_kernel(double* input, double* output, int width, int height, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    double min_distance = 1e30; 
    for (int j = -radius; j <= radius; ++j) {
        for (int i = -radius; i <= radius; ++i) {
            int nx = x + i;
            int ny = y + j;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                double neighbor_distance = input[ny * width + nx];
                min_distance = min(min_distance, neighbor_distance);
            }
        }
    }

    output[y * width + x] = min_distance;
}
void morphological_opening_cuda(double* d_input, double* d_output, int width, int height, int radius) {
    double *d_temp;
    cudaMalloc(&d_temp, width * height * sizeof(double));

    dim3 blockSize(16, 16); 
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    erode_kernel<<<gridSize, blockSize>>>(d_input, d_temp, width, height, radius);
    cudaDeviceSynchronize();

    dilate_kernel<<<gridSize, blockSize>>>(d_temp, d_output, width, height, radius);
    cudaDeviceSynchronize();

    cudaFree(d_temp);
}

__global__ void apply_hysteresis_threshold_kernel(double* data, double* output, int width, int height, double low_threshold, double high_threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    double pixel_distance = data[idx];
    double result_distance = 0.0;

    if (pixel_distance < low_threshold) {
        result_distance = 0.0;
    } else if (pixel_distance > high_threshold) {
        result_distance = 255.0;
    } else {
        result_distance = 128;
    }

    output[idx] = result_distance;
}






__global__ void mask_to_rgb_kernel(double* mask_data, RGB* image_data, RGB* output_data, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    double value = mask_data[idx];
    double new_red = image_data[idx].R + 0.5 * value;
    new_red = min(new_red, 255.0);  

    output_data[idx].R = static_cast<unsigned char>(new_red);
    output_data[idx].G = image_data[idx].G;
    output_data[idx].B = image_data[idx].B;
}

RGBImage apply_mask_to_rgb(Mask& mask, RGBImage& image) {
    int width = mask.get_width();
    int height = mask.get_height();
    RGBImage output_image(width, height);

    int numPixels = width * height;
    size_t bufferSize = numPixels * sizeof(double);

    double *d_mask; 
    RGB *d_output;
    RGB *d_image;
    cudaMalloc(&d_mask, bufferSize);
    cudaMalloc(&d_output, bufferSize);

    cudaMalloc(&d_image, numPixels * sizeof(RGB));
    cudaMemcpy(d_image, image.buffer.data(), numPixels * sizeof(RGB), cudaMemcpyHostToDevice);

    cudaMemcpy(d_mask, mask.buffer.data(), bufferSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, image.buffer.data(), numPixels*sizeof(RGB), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    mask_to_rgb_kernel<<<gridSize, blockSize>>>(d_mask, d_image, d_output, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(output_image.buffer.data(), d_output,numPixels*sizeof(RGB) , cudaMemcpyDeviceToHost);

    cudaFree(d_mask);
    cudaFree(d_output);

    return output_image;
}
    



Mask apply_hysteresis_threshold_cuda(Mask& mask, double low_threshold, double high_threshold) {
    int width = mask.get_width();
    int height = mask.get_height();
    size_t numPixels = width * height;
    size_t bufferSize = numPixels * sizeof(double);

    double *d_input, *d_output;
    cudaMalloc(&d_input, bufferSize);
    cudaMalloc(&d_output, bufferSize);

    cudaMemcpy(d_input, mask.buffer.data(), bufferSize, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    apply_hysteresis_threshold_kernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, low_threshold, high_threshold);
    cudaDeviceSynchronize();

    Mask result_mask(width, height);
    cudaMemcpy(result_mask.buffer.data(), d_output, bufferSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return result_mask;
}



