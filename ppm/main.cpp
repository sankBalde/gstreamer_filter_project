#include "image.hh"
#include "utils.hh"

void print_mask_values(const Mask& mask) {
    for (int y = 0; y < mask.height; ++y) {
        for (int x = 0; x < mask.width; ++x) {
            double distance = mask.get_distance(x, y);
            std::cout << static_cast<int>(distance) << " ";
        }
        std::cout << std::endl;
    }
}
void print_lab_values(const LabImage& lab_image) {
    for (int y = 0; y < lab_image.height; ++y) {
        for (int x = 0; x < lab_image.width; ++x) {
            Lab val = lab_image.buffer[y * lab_image.width + x];
            std::cout << "{" <<val.L << "," << val.a << "," << val.b << "}" << " ";
        }
        std::cout << std::endl;
    }
}




Mask erosion(Mask& mask, int rayon) {
    Mask mask_erosion = mask;
    auto width = mask.width;
    auto height = mask.height;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double min = 255;
            for (int i = -rayon; i <= rayon; i++) {
                for (int j = -rayon; j <= rayon; j++) {
                    if (x + i >= 0 && x + i < width && y + j >= 0 && y + j < height) {
                        double value = mask.get_distance(x + i, y + j);
                        if (value < min) {
                            min = value;
                        }

                    }
                }
            }
            mask_erosion.set_distance(x, y, min);
        }
    }
    return mask_erosion;
}
            
    
  

Mask dilatation(Mask& mask, int rayon) {
    Mask mask_dilatation = mask;
    auto width = mask.width;
    auto height = mask.height;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double max = 0;
            for (int i = -rayon; i <= rayon; i++) {
                for (int j = -rayon; j <= rayon; j++) {
                    if (x + i >= 0 && x + i < width && y + j >= 0 && y + j < height) {
                        double value = mask.get_distance(x + i, y + j);
                        if (value > max) {
                            max = value;
                        }
                        
                        
                    }
                }
            }
            mask_dilatation.set_distance(x, y, max);
        }
    }
    return mask_dilatation;
}
   
  


Mask ouverture_morphologique(Mask& mask, int rayon) {
    mask = erosion(mask, rayon);
  //  mask = erosion(mask, rayon);
    
    mask = dilatation(mask, rayon);



    

    
   
    return mask;
}


    

int main() {
    ImageHandler imageHandler;
    std::string imagePath1 = "frame_rgb";
    std::string imagePath2 = "bg_rgb";
    int image_width = 480;
    int image_height = 360;
    RGBImage rgbImage1 = imageHandler.lireRGBIMAGE(imagePath1, image_width, image_height);
    RGBImage rgbImage2 = imageHandler.lireRGBIMAGE(imagePath2,image_width, image_height);

    if (rgbImage1.width > 0 && rgbImage1.height > 0) {

        LabImage lab_image1 = convertrgb2lab(rgbImage1);
        LabImage lab_image2 = convertrgb2lab(rgbImage2);
        //print_lab_values(lab_image1);

        Mask distance_lab = deltaE_cie76(lab_image1, lab_image2);


        //LabImage opening_lab = morphological_opening(distance_lab, 3);
        
        auto ouverture = ouverture_morphologique(distance_lab, 3);
        
        

        
        

        

        Mask hysteris = apply_hysteresis_threshold(ouverture, 5, 29);
        //print_mask_values(distance_lab);

        RGBImage final = mask_to_rgb(hysteris);
        //print_mask_values(distance_lab);



        //RGBImage final = convertlab2rgb(lab_image1);

        // Exemple de sauvegarde de l'image
        std::string saveImagePath = "final_im.ppm";
        if (imageHandler.savePPM(saveImagePath, final)) {
            std::cout << "Image sauvegardée avec succès." << std::endl;
        } else {
            std::cerr << "Erreur lors de la sauvegarde de l'image." << std::endl;
        }
    }

    /*// Exemple d'utilisation de la conversion RGB vers Lab puis de Lab vers RGB
    RGB rgb_pixel = {100, 150, 200};
    Lab lab_pixel = RGBtoLab(rgb_pixel);
    RGB rgb_pixel_converted = LabtoRGB(lab_pixel);
    std::cout << static_cast<int>(lab_pixel.L) << " " << static_cast<int>(lab_pixel.a) << " " << static_cast<int>(lab_pixel.b) << "\n";



    std::cout << static_cast<int>(rgb_pixel_converted.R) << " " << static_cast<int>(rgb_pixel_converted.G) << " " << static_cast<int>(rgb_pixel_converted.B) << "\n";*/

    return 0;
}
