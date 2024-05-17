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


int main() {
    /*ImageHandler imageHandler;
    std::string imagePath1 = "../frame_rgb";
    std::string imagePath2 = "../bg_rgb";
    int image_width = 480;
    int image_height = 360;
    RGBImage rgbImage1 = imageHandler.lireRGBIMAGE(imagePath1, image_width, image_height);
    RGBImage rgbImage2 = imageHandler.lireRGBIMAGE(imagePath2,image_width, image_height);

    if (rgbImage1.width > 0 && rgbImage1.height > 0) {

        LabImage lab_image1 = convertrgb2lab(rgbImage1);
        LabImage lab_image2 = convertrgb2lab(rgbImage2);
        //print_lab_values(lab_image1);

        Mask distance_lab = deltaE_cie76(lab_image1, lab_image2);


        Mask opening_mask = morphological_opening(distance_lab, 3);

        Mask hysteris = apply_hysteresis_threshold(opening_mask, 4, 30);
        //print_mask_values(distance_lab);

        RGBImage final = mask_to_rgb(hysteris, rgbImage1);
        //RGBImage final = mask_to_rgb(opening_mask);
        //print_mask_values(distance_lab);

        std::string saveImagePath = "../final_im.ppm";
        if (imageHandler.savePPM(saveImagePath, final)) {
            std::cout << "Image sauvegardée avec succès." << std::endl;
        } else {
            std::cerr << "Erreur lors de la sauvegarde de l'image." << std::endl;
        }
    }
    return 0;*/


    /*
    // Création d'un masque de taille 4x4
    Mask mask(4, 4);

    // Initialisation des distances dans le masque
    for (int y = 0; y < mask.get_height(); ++y) {
        for (int x = 0; x < mask.get_width(); ++x) {
            mask.set_distance(x, y, static_cast<double>(x + y)); // Exemple de distance
        }
    }

    Mask hyst = apply_hysteresis_threshold(mask, 1.5, 4.5);

    // Affichage des distances
    for (int y = 0; y < hyst.get_height(); ++y) {
        for (int x = 0; x < hyst.get_width(); ++x) {
            std::cout << hyst.get_distance(x, y) << " ";
        }
        std::cout << "\n";
    }

    return 0;*/
}
