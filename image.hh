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

    RGBImage lireRGBIMAGE(const std::string& nomFichier, int width, int height) {
        int nbre_ligne = width * height;
        std::ifstream fichier(nomFichier);
        RGBImage image(width, height);

        if (fichier) {

            std::string ligne;
            int ligneIndex = 0;

            while (std::getline(fichier, ligne) && ligneIndex < nbre_ligne) { // Lit chaque ligne du fichier et vérifie la limite du nombre de lignes
                std::istringstream iss(ligne); // Crée un flux de chaîne de caractères à partir de la ligne lue
                int valeur; // Variable pour stocker chaque valeur R, G, B du pixel
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

        ppmFile << "P3\n"; // Entête PPM pour un fichier texte
        ppmFile << image.width << " " << image.height << "\n"; // Dimensions de l'image
        ppmFile << "255\n"; // Valeur maximale des composantes de couleur

        for (const auto& pixel : image.buffer) {
            ppmFile << static_cast<int>(pixel.R) << " "; // Composante rouge
            ppmFile << static_cast<int>(pixel.G) << " "; // Composante verte
            ppmFile << static_cast<int>(pixel.B) << " "; // Composante bleue
        }

        ppmFile.close();
        return true;
    }
};