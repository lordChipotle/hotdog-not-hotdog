// DataLoader.cpp
#include "DataLoader.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <string>

struct ImageData {
    cv::Mat image;
    int label;
};

std::vector<ImageData> loadData(const std::string& csvPath, const std::string& imageFolder) {
    std::vector<ImageData> data;
    std::ifstream file(csvPath);
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<std::string> tokens;

        while (std::getline(lineStream, cell, ',')) {
            tokens.push_back(cell);
        }

        std::string imagePath = imageFolder + "/" + tokens[0];
        int label = std::stoi(tokens[1]);

        cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
        cv::resize(image, image, cv::Size(width, height));  // Resize to match the input size of the network

        data.push_back({image, label});
    }

    return data;
}

void preprocessAndTransfer(std::vector<ImageData>& data, float* d_input) {
    std::vector<float> hostInputBuffer(batch_size * channels * height * width);

    // Convert images to the format expected by the network (e.g., normalize, flatten, etc.)
    for (size_t i = 0; i < data.size(); ++i) {
        cv::Mat imageFloat;
        data[i].image.convertTo(imageFloat, CV_32FC3);
        std::memcpy(hostInputBuffer.data() + i * channels * height * width, imageFloat.data, channels * height * width * sizeof(float));
    }

    // Normalize
    /for (auto& pixel : hostInputBuffer) pixel = normalizePixel(pixel);

    // Transfer data to the GPU
    cudaMemcpy(d_input, hostInputBuffer.data(), batch_size * channels * height * width * sizeof(float), cudaMemcpyHostToDevice);
}

