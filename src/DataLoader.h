// DataLoader.h
#ifndef DATALOADER_H
#define DATALOADER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct ImageData {
    cv::Mat image;
    int label;
};

std::vector<ImageData> loadData(const std::string& csvPath, const std::string& imageFolder);
void preprocessAndTransfer(std::vector<ImageData>& data, float* d_input);

#endif
