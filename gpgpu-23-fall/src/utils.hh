#pragma once
#include "image.hh"
#include <cmath>
#include <vector>

#include <cstdint>
#include <stdexcept>
#include <omp.h>
#include <iostream>
#include <cstdlib> 
#include <string>  
#include <optional> 


std::vector<RGB> uint8_to_rgb(const uint8_t* buffer, int w, int h) {
    size_t num_pixels = w * h;
    std::vector<RGB> rgb_data(num_pixels);
    
    for (size_t i = 0; i < num_pixels; ++i) {
        rgb_data[i].R = buffer[i * 3];
        rgb_data[i].G = buffer[i * 3 + 1];
        rgb_data[i].B = buffer[i * 3 + 2];
    }

    return rgb_data;
}


uint8_t* rgb_to_uint8(const std::vector<RGB>& rgb_data) {
    size_t num_pixels = rgb_data.size();
    uint8_t* buffer = new uint8_t[num_pixels * 3];

    //#pragma omp parallel for
    for (size_t i = 0; i < num_pixels; ++i) {
        buffer[i * 3] = rgb_data[i].R;
        buffer[i * 3 + 1] = rgb_data[i].G;
        buffer[i * 3 + 2] = rgb_data[i].B;
    }

    return buffer;
}



Lab RGBtoLab(const RGB& rgb) {

    double R = rgb.R / 255.0;
    double G = rgb.G / 255.0;
    double B = rgb.B / 255.0;


    if (R > 0.04045) {
        R = std::pow((R + 0.055) / 1.055, 2.4);
    } else {
        R = R / 12.92;
    }
    if (G > 0.04045) {
        G = std::pow((G + 0.055) / 1.055, 2.4);
    } else {
        G = G / 12.92;
    }
    if (B > 0.04045) {
        B = std::pow((B + 0.055) / 1.055, 2.4);
    } else {
        B = B / 12.92;
    }


    double X = R * 100 * 0.4124 + G * 100 * 0.3576 + B * 100 * 0.1805;
    double Y = R * 100 * 0.2126 + G * 100 * 0.7152 + B * 100 * 0.0722;
    double Z = R * 100 * 0.0193 + G * 100 * 0.1192 + B * 100 * 0.9505;


    X /= 95.047;
    Y /= 100.0;
    Z /= 108.883;


    if (X > 0.008856) {
        X = std::pow(X, 1.0 / 3.0);
    } else {
        X = (7.787 * X) + (16.0 / 116.0);
    }
    if (Y > 0.008856) {
        Y = std::pow(Y, 1.0 / 3.0);
    } else {
        Y = (7.787 * Y) + (16.0 / 116.0);
    }
    if (Z > 0.008856) {
        Z = std::pow(Z, 1.0 / 3.0);
    } else {
        Z = (7.787 * Z) + (16.0 / 116.0);
    }

    double L = (116.0 * Y) - 16.0;
    double a = 500.0 * (X - Y);
    double b = 200.0 * (Y - Z);

    return {L, a, b};
}

RGB LabtoRGB(const Lab& lab) {

    double ref_X = 95.047;
    double ref_Y = 100.000;
    double ref_Z = 108.883;


    double var_Y = (lab.L + 16.0) / 116.0;
    double var_X = lab.a / 500.0 + var_Y;
    double var_Z = var_Y - lab.b / 200.0;


    if (std::pow(var_Y, 3.0) > 0.008856) {
        var_Y = std::pow(var_Y, 3.0);
    } else {
        var_Y = (var_Y - 16.0 / 116.0) / 7.787;
    }
    if (std::pow(var_X, 3.0) > 0.008856) {
        var_X = std::pow(var_X, 3.0);
    } else {
        var_X = (var_X - 16.0 / 116.0) / 7.787;
    }
    if (std::pow(var_Z, 3.0) > 0.008856) {
        var_Z = std::pow(var_Z, 3.0);
    } else {
        var_Z = (var_Z - 16.0 / 116.0) / 7.787;
    }


    double X = var_X * ref_X / 100.0;
    double Y = var_Y * ref_Y / 100.0;
    double Z = var_Z * ref_Z / 100.0;

    double var_R = X *  3.2406 + Y * -1.5372 + Z * -0.4986;
    double var_G = X * -0.9689 + Y *  1.8758 + Z *  0.0415;
    double var_B = X *  0.0557 + Y * -0.2040 + Z *  1.0570;


    if (var_R > 0.0031308) {
        var_R = 1.055 * std::pow(var_R, 1.0 / 2.4) - 0.055;
    } else {
        var_R = 12.92 * var_R;
    }
    if (var_G > 0.0031308) {
        var_G = 1.055 * std::pow(var_G, 1.0 / 2.4) - 0.055;
    } else {
        var_G = 12.92 * var_G;
    }
    if (var_B > 0.0031308) {
        var_B = 1.055 * std::pow(var_B, 1.0 / 2.4) - 0.055;
    } else {
        var_B = 12.92 * var_B;
    }


    uint8_t R = static_cast<uint8_t>(std::round(var_R * 255.0));
    uint8_t G = static_cast<uint8_t>(std::round(var_G * 255.0));
    uint8_t B = static_cast<uint8_t>(std::round(var_B * 255.0));

    return {R, G, B};
}
Mask dilate(Mask& mask, int radius) {
    Mask dilated_mask(mask.width, mask.height);

    for (int y = 0; y < mask.height; ++y) {
        for (int x = 0; x < mask.width; ++x) {
            double max_distance = mask.get_distance(x, y);

            for (int j = -radius; j <= radius; ++j) {
                for (int i = -radius; i <= radius; ++i) {
                    int neighbor_x = x + i;
                    int neighbor_y = y + j;
                    if (neighbor_x >= 0 && neighbor_x < mask.width && neighbor_y >= 0 && neighbor_y < mask.height) {
                        double neighbor_distance = mask.get_distance(neighbor_x, neighbor_y);
                        max_distance = std::max(max_distance, neighbor_distance);
                    }
                }
            }

            dilated_mask.set_distance(x, y, max_distance);
        }
    }

    return dilated_mask;
}

Mask erode(Mask& mask, int radius) {
    Mask eroded_mask(mask.width, mask.height);
    for (int y = 0; y < mask.height; ++y) {
        for (int x = 0; x < mask.width; ++x) {
            double min_distance = mask.get_distance(x, y);

            for (int j = -radius; j <= radius; ++j) {
                for (int i = -radius; i <= radius; ++i) {
                    int neighbor_x = x + i;
                    int neighbor_y = y + j;
                    if (neighbor_x >= 0 && neighbor_x < mask.width && neighbor_y >= 0 && neighbor_y < mask.height) {
                        double neighbor_distance = mask.get_distance(neighbor_x, neighbor_y);
                        min_distance = std::min(min_distance, neighbor_distance);
                    }
                }
            }

            eroded_mask.set_distance(x, y, min_distance);
        }
    }

    return eroded_mask;
}

Mask morphological_opening(Mask& mask, int radius) {
    Mask eroded_mask = erode(mask, radius);
    Mask opened_mask = dilate(eroded_mask, radius);
    return opened_mask;
}




Mask apply_hysteresis_threshold(Mask& mask, int low_threshold, int high_threshold) {
    Mask result_mask(mask.width, mask.height);

    for (int y = 0; y < mask.height; ++y) {
        for (int x = 0; x < mask.width; ++x) {
            double pixel_distance = mask.get_distance(x, y);
            double result_distance = 0.0;

            if (pixel_distance < low_threshold) {
                result_distance = 0.0;
            } else if (pixel_distance > high_threshold) {
                result_distance = 255.0;
            } else {
                result_distance = 129.0;
            }
            result_mask.set_distance(x, y, result_distance);
        }
    }

    return result_mask;
}



RGBImage mask_to_rgb(Mask& mask, RGBImage& image) {
    RGBImage rgb_image(mask.get_width(), mask.get_height());

    for (int y = 0; y < mask.get_height(); ++y) {
        for (int x = 0; x < mask.get_width(); ++x) {
            double value = mask.get_distance(x, y);

            double new_red = image.buffer[y * image.width + x].R + 0.5 * value;

            new_red = std::min(new_red, 255.0);

            rgb_image.buffer[y * rgb_image.width + x] = {static_cast<u_int8_t>(new_red), image.buffer[y * image.width + x].G, image.buffer[y * image.width + x].B};
        }
    }

    return rgb_image;
}
RGBImage mask_to_rgb(Mask& mask) {
    RGBImage rgb_image(mask.width, mask.height);

    for (int y = 0; y < mask.height; ++y) {
        for (int x = 0; x < mask.width; ++x) {
            double value = mask.buffer[y * rgb_image.width + x];
            rgb_image.buffer[y * rgb_image.width + x] = {static_cast<u_int8_t>(value), static_cast<u_int8_t>(value), static_cast<u_int8_t>(value)};
        }
    }
    return rgb_image;
}



double deltaE_cie76(const Lab& lab1, const Lab& lab2) {
    double delta_L = lab1.L - lab2.L;
    double delta_a = lab1.a - lab2.a;
    double delta_b = lab1.b - lab2.b;

    return std::sqrt(delta_L * delta_L + delta_a * delta_a + delta_b * delta_b);
}

Mask deltaE_cie76(LabImage& image1, LabImage& image2) {
    if (image1.width != image2.width || image1.height != image2.height) {
        std::cout << "width: " << image1.width << " " << image2.width << " height " << image1.height<<" " << image2.height << "\n";
        throw std::runtime_error("Les dimensions des images ne correspondent pas.");
    }

    Mask deltaE_map(image1.width, image1.height);

    for (int y = 0; y < image1.height; ++y) {
        for (int x = 0; x < image1.width; ++x) {
            const Lab& lab1 = image1.get_pixel(x, y);
            const Lab& lab2 = image2.get_pixel(x, y);
            double deltaE = deltaE_cie76(lab1, lab2);
            deltaE_map.set_distance(x, y, deltaE);
        }
    }

    return deltaE_map;
}


RGBImage convertlab2rgb(LabImage& lab_image) {
    RGBImage rgb_image(lab_image.width, lab_image.height);
    for (int y = 0; y < lab_image.height; ++y) {
        for (int x = 0; x < lab_image.width; ++x) {
            const Lab& lab = lab_image.get_pixel(x, y);
            const RGB& rgb = LabtoRGB(lab); 
            rgb_image.buffer[y * lab_image.width + x] = rgb;
        }
    }

    return rgb_image;
}

LabImage convertrgb2lab(RGBImage& rgb_image) {
    LabImage lab_image(rgb_image.width, rgb_image.height);
    //#pragma omp parallel for
    for (int y = 0; y < rgb_image.height; ++y) {
        for (int x = 0; x < rgb_image.width; ++x) {
            const RGB& rgb = rgb_image.buffer[y * rgb_image.width + x];
            const Lab& lab = RGBtoLab(rgb);
            lab_image.get_pixel(x, y) = lab;
        }
    }

    return lab_image;
}


RGBImage averageRGBImages(const std::vector<RGBImage>& images) {
    if (images.empty()) {
        return RGBImage(); 
    }
    
    int width = images[0].width;
    int height = images[0].height;

    for (const auto& img : images) {
        if (img.width != width || img.height != height) {
            std::cerr << "Les images doivent avoir les mêmes dimensions." << std::endl;
            return RGBImage();
        }
    }

    std::vector<int> sumR(width * height, 0), sumG(width * height, 0), sumB(width * height, 0);
    int numImages = images.size();

    for (const auto& img : images) {
        for (int i = 0; i < width * height; ++i) {
            sumR[i] += img.buffer[i].R;
            sumG[i] += img.buffer[i].G;
            sumB[i] += img.buffer[i].B;
        }
    }

    RGBImage result(width, height);
    for (int i = 0; i < width * height; ++i) {
        result.buffer[i].R = sumR[i] / numImages;
        result.buffer[i].G = sumG[i] / numImages;
        result.buffer[i].B = sumB[i] / numImages;
    }

    return result;
}



std::optional<int> getEnvAsInt(const std::string& env_var) {
    char* value = getenv(env_var.c_str());
    if (value != nullptr) {
        try {
            return std::stoi(value);
        } catch (const std::invalid_argument& e) {
            std::cerr << "La valeur de '" << env_var << "' n'est pas un entier valide." << std::endl;
        } catch (const std::out_of_range& e) {
            std::cerr << "La valeur de '" << env_var << "' est hors limite pour un int." << std::endl;
        }
    } else {
        std::cerr << "La variable d'environnement '" << env_var << "' n'existe pas." << std::endl;
    }
    return {}; 
}


