#pragma once

#include <fstream>
#include <iostream>
#include <vector>


struct RGB {
    uint8_t R;
    uint8_t G;
    uint8_t B;
};

struct Lab {
    double L;
    double a;
    double b;
};

class Mask {
public:
    // Constructeur par défaut
    Mask() : width(0), height(0) {}

    // Constructeur avec spécification de la taille
    Mask(int sx, int sy) : width(sx), height(sy) {
        buffer.resize(width * height);
    }

    // Accesseur pour obtenir la distance d'un pixel
    double get_distance(int x, int y) const {
        return buffer[y * width + x];
    }

    // Mutateur pour définir la distance d'un pixel
    void set_distance(int x, int y, double distance) {
        buffer[y * width + x] = distance;
    }

    // Accesseur pour obtenir la largeur de la masque
    int get_width() const {
        return width;
    }

    // Accesseur pour obtenir la hauteur de la masque
    int get_height() const {
        return height;
    }


    int width; // Largeur de la masque
    int height; // Hauteur de la masque
    std::vector<double> buffer; // Buffer de distances
};

class LabImage {
public:

    LabImage() : width(0), height(0) {}


    LabImage(int sx, int sy) : width(sx), height(sy) {
        buffer.resize(width * height);
    }


    Lab& get_pixel(int x, int y) {
        return buffer[y * width + x];
    }

public:
    int width;
    int height;
    std::vector<Lab> buffer;
};

class RGBImage {
public:
    RGBImage() : width(0), height(0) {}

    RGBImage(int sx, int sy) : width(sx), height(sy) {
        buffer.resize(width * height);
    }

public:
    int width; // Largeur de l'image en pixels
    int height; // Hauteur de l'image en pixels
    std::vector<RGB> buffer; // Buffer de pixels
};

class ImageHandler {
public:
    RGBImage load_image(const std::string& imagePath, int image_width, int image_height) {
        std::ifstream imageFile(imagePath, std::ios::binary);

        if (!imageFile.is_open()) {
            std::cerr << "Erreur lors de l'ouverture du fichier." << std::endl;
            return RGBImage(); // Retourne une image vide en cas d'erreur
        }

        imageFile.seekg(0, std::ios::end);
        std::streampos fileSize = imageFile.tellg();
        imageFile.seekg(0, std::ios::beg);

        std::vector<char> buffer(fileSize);
        imageFile.read(buffer.data(), fileSize);

        imageFile.close();

        if (!imageFile) {
            std::cerr << "Erreur lors de la lecture du fichier." << std::endl;
            return RGBImage(); // Retourne une image vide en cas d'erreur
        }


        int imageSize = fileSize / sizeof(RGB);
        int width = static_cast<int>(std::sqrt(imageSize));
        int height = imageSize / width;

        // Convertir le buffer en image RGB
        RGBImage rgbImage(image_width, image_height);
        std::memcpy(rgbImage.buffer.data(), buffer.data(), buffer.size());

        return rgbImage;
    }


    bool save_image(const std::string& imagePath, const RGBImage& rgbImage) {
        std::ofstream imageFile(imagePath, std::ios::binary);

        if (!imageFile.is_open()) {
            std::cerr << "Erreur lors de l'ouverture du fichier pour l'écriture." << std::endl;
            return false;
        }

        imageFile.write(reinterpret_cast<const char*>(rgbImage.buffer.data()), rgbImage.buffer.size() * sizeof(RGB));

        if (!imageFile) {
            std::cerr << "Erreur lors de l'écriture de l'image dans le fichier." << std::endl;
            return false;
        }

        imageFile.close();
        return true;
    }
};