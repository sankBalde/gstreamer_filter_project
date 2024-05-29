#include "filter_impl.h"
#include <chrono>
#include <thread>
#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>
#include <thread>
#include "image.hh"
#include "utils.hh"

static std::vector<uint8_t> global_buffer;

extern "C" {
    void filter_impl(uint8_t* buffer, int width, int height, int stride, int pixel_stride)
    {
        size_t buffer_size = height * stride;

        if (global_buffer.empty())
        {
            global_buffer.assign(buffer, buffer + buffer_size);
            return;
        }

        std::vector<RGB> prev_rgb_image_vect = uint8_to_rgb(global_buffer.data(), width, height);
        std::vector<RGB> new_rgb_image_vect = uint8_to_rgb(buffer, width, height);

        RGBImage rgbImage1(width, height);
        RGBImage rgbImage2(width, height);
        rgbImage1.buffer = std::move(prev_rgb_image_vect);
        rgbImage2.buffer = std::move(new_rgb_image_vect);

        LabImage lab_image1 = convertrgb2lab(rgbImage1);
        LabImage lab_image2 = convertrgb2lab(rgbImage2);

        Mask distance_lab = deltaE_cie76(lab_image1, lab_image2);
        Mask opening_mask = morphological_opening(distance_lab, 3);
        Mask hysteris = apply_hysteresis_threshold(opening_mask, 4, 30);

        RGBImage final = mask_to_rgb(hysteris, rgbImage2);



        uint8_t* final_buffer_ptr = rgb_to_uint8(final.buffer);

        if (final_buffer_ptr == nullptr) {
            std::cerr << "Erreur: rgb_to_uint8 a retourné un pointeur nul." << std::endl;
            return;
        }

        //global_buffer.assign(buffer, buffer + buffer_size);

        for (size_t i = 0; i < buffer_size; ++i) {
            buffer[i] = final_buffer_ptr[i];
        }

        delete[] final_buffer_ptr;



        using namespace std::chrono_literals;


    }
}
