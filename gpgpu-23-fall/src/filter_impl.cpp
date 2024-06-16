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
static std::vector<RGBImage> background_images;
static int cpt_frame = 0;

static int opening_size;
static int th_low;
static int th_high;
static int bg_number_frame;


extern "C" {
    void filter_impl(uint8_t* buffer, int width, int height, int stride, int pixel_stride)
    {



      
        
        cpt_frame++;
        size_t buffer_size = height * stride;

        if (global_buffer.empty())
        {
            global_buffer.assign(buffer, buffer + buffer_size);
            std::string opening_size_var = "OPENING_SIZE_ENV";
            std::string th_low_var = "TH_LOW_ENV";
            std::string th_high_var = "TH_HIGH_ENV";
            std::string bg_number_frame_var = "BG_NUMBER_FRAME_ENV";
            opening_size = getEnvAsInt(opening_size_var).value_or(3);
            th_low = getEnvAsInt(th_low_var).value_or(4);
            th_high = getEnvAsInt(th_high_var).value_or(30);
            bg_number_frame = getEnvAsInt(bg_number_frame_var).value_or(10);
            return;
        }

        
        std::vector<RGB> prev_rgb_image_vect = uint8_to_rgb(global_buffer.data(), width, height);
        std::vector<RGB> new_rgb_image_vect = uint8_to_rgb(buffer, width, height);


        RGBImage rgbImage1(width, height);
        RGBImage rgbImage2(width, height);
        rgbImage1.buffer = std::move(prev_rgb_image_vect);
        rgbImage2.buffer = std::move(new_rgb_image_vect);

        background_images.push_back(rgbImage2);
        if (cpt_frame == bg_number_frame-1)
        {
            cpt_frame = 0;
            background_images.push_back(rgbImage1);
            RGBImage average = averageRGBImages(background_images);
            background_images.clear();
            uint8_t* average_buffer_ptr = rgb_to_uint8(average.buffer);
            global_buffer.assign(average_buffer_ptr, average_buffer_ptr + buffer_size);
        }
            

        LabImage lab_image1 = convertrgb2lab(rgbImage1);
        LabImage lab_image2 = convertrgb2lab(rgbImage2);

        Mask distance_lab = deltaE_cie76(lab_image1, lab_image2);
        Mask opening_mask = morphological_opening(distance_lab, opening_size);
        Mask hysteris = apply_hysteresis_threshold(opening_mask, th_low, th_high);

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
