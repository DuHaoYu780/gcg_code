#include <opencv2/opencv.hpp>
#include <iostream>
#include "ORBextractor.h"
#include <chrono>
using namespace cv;
using namespace std;

// This function extracts ORB features and displays the number of features
void extractAndShowORBFeatures(const cv::Mat& image, const std::string& windowName) {
    // Create ORB feature detector
    ORB_SLAM3::ORBextractor mpORBextractorLeft(3000, 1.2, 8, 20, 8);
    std::vector<cv::KeyPoint> mvKeys;
    cv::Mat mDescriptors;
    std::vector<int> vLapping = {0, 1};
    // Detect features and compute descriptors
    mpORBextractorLeft.operator()(image, cv::Mat(), mvKeys, mDescriptors, vLapping);
    // Draw feature points
    cv::Mat imageWithKeypoints;
    cv::drawKeypoints(image, mvKeys, imageWithKeypoints);
    // Display the number of ORB features
    std::cout << "Number of ORB features in " << windowName << ": " << mvKeys.size() << std::endl;
    // Show image with feature points
    cv::imshow(windowName, imageWithKeypoints);
}

// Gamma correction
void gammaCorrection(const Mat& input, Mat& output, double gamma) {
    // Ensure gamma value is reasonable
    if (gamma <= 0) gamma = 1.0;
    // Create lookup table
    Mat lut(1, 256, CV_8U);
    uchar* p = lut.ptr();
    for (int i = 0; i < 256; ++i) {
        // Gamma correction formula: output = input^(gamma) * 255
        p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
    }
    // Apply lookup table
    LUT(input, lut, output);
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        cerr << endl << "Usage: ./clahe path_to_image " << endl;
        return 1;
    }
    // Read image
    cv::Mat readimage = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (readimage.empty()) {
        std::cerr << "Could not open or find the image" << std::endl;
        return -1;
    }
    
    cv::Size targetSize(640, 480);
    cv::Mat image;
    cv::resize(readimage, image, targetSize);
    
    cv::imshow("Original", image);
    extractAndShowORBFeatures(image, "Original Image with ORB Features");

// 1. Gamma Correction
    auto start1 = std::chrono::high_resolution_clock::now();
        cv::Mat enhanced;
        double gamma = 0.6; // Adjust this gamma value (>1 darkens, <1 brightens)
        // Perform Gamma correction
        gammaCorrection(image, enhanced, gamma);
    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count();
    std::cout << "Gamma Correction algorithm running time: " << duration1 << " microseconds" << std::endl;
    
    cv::imshow("Gamma Correction", enhanced);
    extractAndShowORBFeatures(enhanced, "Gamma Correction Image with ORB Features");
    
// 2. CLAHE Enhancement
    auto start2 = std::chrono::high_resolution_clock::now();
        // Create CLAHE object
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        // Set contrast limit
        clahe->setClipLimit(2.0);
        // Set tile grid size
        clahe->setTilesGridSize(cv::Size(8, 8));
        // Apply CLAHE algorithm
        cv::Mat claheImage;
        clahe->apply(image, claheImage);
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();
    std::cout << "CLAHE Enhancement algorithm running time: " << duration2 << " microseconds" << std::endl;
    
    cv::imshow("CLAHE Enhancement", claheImage);
    extractAndShowORBFeatures(claheImage, "CLAHE Enhancement Image with ORB Features");
       
// 3. Gamma + CLAHE Enhancement
{
    auto start3 = std::chrono::high_resolution_clock::now();
        cv::Mat enhanced1;
        
        // Perform Gamma correction
        gammaCorrection(image, enhanced1, gamma);
        cv::Ptr<cv::CLAHE> clahe1 = cv::createCLAHE();
        // Set contrast limit
        clahe1->setClipLimit(2.0);
        // Set tile grid size
        clahe1->setTilesGridSize(cv::Size(8, 8));
        cv::Mat gammaEnhanced;
        clahe1->apply(enhanced1, gammaEnhanced);
    auto end3 = std::chrono::high_resolution_clock::now();
    auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(end3 - start3).count();
    std::cout << "Gamma + CLAHE Enhancement algorithm running time: " << duration3 << " microseconds" << std::endl;
    
    cv::imshow("Gamma + CLAHE Enhancement", gammaEnhanced);
    extractAndShowORBFeatures(gammaEnhanced, "Gamma + CLAHE Enhancement Image with ORB Features");
}

// 4. Gamma + CLAHE + Gaussian Blur
{
    auto start4 = std::chrono::high_resolution_clock::now();
    
        cv::Mat gammaEnhanced1;
        gammaCorrection(image, gammaEnhanced1, gamma);
        cv::Ptr<cv::CLAHE> clahe2 = cv::createCLAHE();
        // Set contrast limit
        clahe2->setClipLimit(2.0);
        // Set tile grid size
        clahe2->setTilesGridSize(cv::Size(8, 8));
        cv::Mat claheEnhanced1;
        clahe2->apply(gammaEnhanced1, claheEnhanced1);
        // Apply Gaussian blur
        cv::Mat gaussian_blurred;
        cv::GaussianBlur(claheEnhanced1, gaussian_blurred, cv::Size(5, 5), 0);
    auto end4 = std::chrono::high_resolution_clock::now();
    auto duration4 = std::chrono::duration_cast<std::chrono::microseconds>(end4 - start4).count();
    std::cout << "Gamma + CLAHE Enhancement + Gaussian Blur algorithm running time: " << duration4 << " microseconds" << std::endl;
    
    cv::imshow("Gamma + CLAHE Enhancement + Gaussian Blur", gaussian_blurred);
    extractAndShowORBFeatures(gaussian_blurred, "Gamma + CLAHE Enhancement + Gaussian Blur Image with ORB Features");
}

    // Wait for key press
    cv::waitKey(0);
    // Close all windows
    cv::destroyAllWindows();
    return 0;
}
