#include "DataLoader.h"
#include <cudnn.h>
#include <iostream>
#include <vector>

#define CHECK_CUDNN(status) { \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "cuDNN error at line " << __LINE__ << ": " \
                  << cudnnGetErrorString(status) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

int main() {
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    int batch_size = 64;
    int channels = 3;
    int height = 32;
    int width = 32;

    // Initialize the input tensor descriptor
    cudnnTensorDescriptor_t input_descriptor;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, channels, height, width));

    int out_channels = 6;
    int kernel_size = 5;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnTensorDescriptor_t conv_output_desc;

    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out_channels, channels, kernel_size, kernel_size));

    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc, 2, 2, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    int conv_output_height, conv_output_width;
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(conv_desc, input_descriptor, filter_desc, &batch_size, &out_channels, &conv_output_height, &conv_output_width));

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&conv_output_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(conv_output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, out_channels, conv_output_height, conv_output_width));

    // Allocate memory for the convolution output
    float* d_conv_output;
    cudaMalloc(&d_conv_output, batch_size * out_channels * conv_output_height * conv_output_width * sizeof(float));


    // Perform the convolution forward pass
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha, input_descriptor, d_input, filter_desc, d_filter, conv_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, nullptr, 0, &beta, conv_output_desc, d_conv_output));

    // Initialize and apply the activation function
    cudnnActivationDescriptor_t activation_desc;
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&activation_desc));
    CHECK_CUDNN(cudnnSetActivationDescriptor(activation_desc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0));
    
    CHECK_CUDNN(cudnnActivationForward(cudnn, activation_desc, &alpha, conv_output_desc, d_conv_output, &beta, conv_output_desc, d_conv_output));

    // Pooling layer
    cudnnPoolingDescriptor_t pooling_desc;
    CHECK_CUDNN(cudnnCreatePoolingDescriptor(&pooling_desc));
    CHECK_CUDNN(cudnnSetPooling2dDescriptor(pooling_desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 0, 0, 2, 2));

    int pool_output_height = conv_output_height / 2;
    int pool_output_width = conv_output_width / 2;
    cudnnTensorDescriptor_t pool_output_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&pool_output_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(pool_output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, out_channels, pool_output_height, pool_output_width));

    // Allocate memory for the pooling output
    float* d_pool_output;
    cudaMalloc(&d_pool_output, batch_size * out_channels * pool_output_height * pool_output_width * sizeof(float));

    // Perform the pooling forward pass
    CHECK_CUDNN(cudnnPoolingForward(cudnn, pooling_desc, &alpha, conv_output_desc, d_conv_output, &beta, pool_output_desc, d_pool_output));

    // Fully connected layer (omitted details for brevity)

    // Clean up
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroyTensorDescriptor(conv_output_desc);
    cudnnDestroyActivationDescriptor(activation_desc);
    cudnnDestroyPoolingDescriptor(pooling_desc);
    cudnnDestroyTensorDescriptor(pool_output_desc);
    cudaFree(d_conv_output);
    cudaFree(d_pool_output);
    cudnnDestroy(cudnn);

    return 0;
}
