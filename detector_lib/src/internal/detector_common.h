#ifndef DETECTOR_COMMON_H
#define DETECTOR_COMMON_H

// 标准库
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <fstream>

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

// RKNN (使用项目内的头文件)
extern "C" {
#include "rknn_api.h"
#include "detector_common_types.h"
// 暂时移除可能包含RGA依赖的头文件
// #include "image_utils.h"
#include "detector_file_utils.h"
}

// 注意: 暂时移除复杂的项目内部头文件依赖
// 这些头文件可能包含RGA等不必要的依赖
// 我们的简化版本暂时使用模拟数据进行演示
// #include "pose_yolov8.h"
// #include "pose_postprocess.h"
// #include "pose_letterbox_utils.h"
// #include "rim_basketball_postprocess.h"

// C++11兼容性处理
#if __cplusplus < 201402L
namespace std {
    template<typename T, typename... Args>
    std::unique_ptr<T> make_unique(Args&&... args) {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }
}
#endif

namespace detector {
namespace internal {

// 简化版本 - 移除日志系统

// 内部工具函数
int64_t get_current_time_us();
bool file_exists(const std::string& path);
std::string get_file_extension(const std::string& path);

// RKNN包装结构
struct RknnContext {
    rknn_context ctx = 0;
    rknn_input_output_num io_num = {};
    rknn_tensor_attr* input_attrs = nullptr;
    rknn_tensor_attr* output_attrs = nullptr;
    int model_width = 0;
    int model_height = 0;
    int model_channel = 0;
    bool is_initialized = false;
    bool is_quant = false;  // 添加量化标志
    int assigned_npu_core = -1;  // 记录分配的NPU核心 (-1=auto, 0=core0, 1=core1)
    
    ~RknnContext();
    bool init_from_file(const std::string& model_path);
    void cleanup();
};

// 零拷贝内存管理
struct ZeroCopyMemory {
    RknnContext* rknn_ctx = nullptr;
    rknn_tensor_mem* input_mem = nullptr;
    std::vector<rknn_tensor_mem*> output_mems;
    rknn_tensor_attr input_attr = {};
    std::vector<rknn_tensor_attr> output_attrs;
    bool is_initialized = false;
    
    ~ZeroCopyMemory();
    bool init(RknnContext* ctx);
    void cleanup();
};

} // namespace internal
} // namespace detector

#endif // DETECTOR_COMMON_H