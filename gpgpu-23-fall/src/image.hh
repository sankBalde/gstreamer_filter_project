#pragma once

#include <fstream>
#include <iostream>
#include <vector>

#include <string>
#include <sstream>


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

    Mask() : width(0), height(0) {}

    Mask(int sx, int sy) : width(sx), height(sy) {
        buffer.resize(width * height);
    }

    double get_distance(int x, int y) const {
        return buffer[y * width + x];
    }

    void set_distance(int x, int y, double distance) {
        buffer[y * width + x] = distance;
    }

    int get_width() const {
        return width;
    }

    int get_height() const {
        return height;
    }


    int width;
    int height;
    std::vector<double> buffer;
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

    int width;
    int height;
    std::vector<RGB> buffer;
};

class ImageHandler {
public:

    RGBImage lireRGBIMAGE(const std::string& nomFichier, int width, int height) {
        int nbre_ligne = width * height;
        std::ifstream fichier(nomFichier);
        RGBImage image(width, height);

        if (fichier) {

            std::string ligne;
            int ligneIndex = 0;

            while (std::getline(fichier, ligne) && ligneIndex < nbre_ligne) {
                std::istringstream iss(ligne);
                int valeur;
                u_int8_t R, G, B;
                for (int j = 0; j < 3 && iss >> valeur; ++j) {
                    if (j == 0){
                        R = valeur;
                    }
                    else if (j == 1) {
                        G = valeur;
                    }
                    else {
                        B = valeur;
                    }

                }
                image.buffer[ligneIndex] = {R, G, B};

                if (!iss.eof()) {
                    std::cerr << "Erreur lors de la lecture du fichier." << std::endl;
                    return RGBImage();
                }

                ++ligneIndex;
            }
            fichier.close();
        }
        else {
            std::cerr << "Erreur lors de l'ouverture du fichier." << std::endl;
        }

        return image;
    }

    bool savePPM(const std::string& filename, RGBImage& image) {
        std::ofstream ppmFile(filename);
        if (!ppmFile) {
            std::cerr << "Erreur lors de l'ouverture du fichier pour l'écriture." << std::endl;
            return false;
        }

        ppmFile << "P3\n";
        ppmFile << image.width << " " << image.height << "\n";
        ppmFile << "255\n";

        for (const auto& pixel : image.buffer) {
            ppmFile << static_cast<int>(pixel.R) << " ";
            ppmFile << static_cast<int>(pixel.G) << " ";
            ppmFile << static_cast<int>(pixel.B) << " ";
        }

        ppmFile.close();
        return true;
    }
};