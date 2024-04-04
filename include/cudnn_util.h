// cudnn_util.h
#ifndef CUDNN_UTIL_H
#define CUDNN_UTIL_H

#include <cudnn.h>
#include <iostream>

#define CHECK_CUDNN(status) { \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "cuDNN error at line " << __LINE__ << ": " \
                  << cudnnGetErrorString(status) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#endif
