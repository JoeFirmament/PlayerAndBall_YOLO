#include "detector_common.h"
#include <sys/time.h>
#include <sys/stat.h>

namespace detector {
namespace internal {

int64_t get_current_time_us() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

bool file_exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

std::string get_file_extension(const std::string& path) {
    size_t dot_pos = path.find_last_of('.');
    if (dot_pos == std::string::npos) return "";
    return path.substr(dot_pos + 1);
}

// RknnContext 实现
RknnContext::~RknnContext() {
    cleanup();
}

bool RknnContext::init_from_file(const std::string& model_path) {
    if (is_initialized) {
        return true;
    }
    
    
    
    if (!file_exists(model_path)) {
        printf("❌ 错误: 找不到模型文件: %s\n", model_path.c_str());
        return false;
    }
    
    
    int ret = rknn_init(&ctx, (char*)model_path.c_str(), 0, 0, nullptr);
    
    if (ret < 0) {
        printf("❌ 错误: RKNN初始化失败 (错误码: %d)\n", ret);
        return false;
    }
    
    
    
    // 获取输入输出信息
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        printf("❌ 错误: 无法查询模型输入输出数量 (错误码: %d)\n", ret);
        cleanup();
        return false;
    }
    
    
    // 获取输入属性
    input_attrs = new rknn_tensor_attr[io_num.n_input];
    memset(input_attrs, 0, io_num.n_input * sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attrs[i], sizeof(rknn_tensor_attr));
        if (ret < 0) {
        printf("❌ 错误: 无法查询输入属性 %d (错误码: %d)\n", i, ret);
            cleanup();
            return false;
        }
    }
    
    // 获取输出属性
    output_attrs = new rknn_tensor_attr[io_num.n_output];
    memset(output_attrs, 0, io_num.n_output * sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr));
        if (ret < 0) {
        printf("❌ 错误: 无法查询输出属性 %d (错误码: %d)\n", i, ret);
            cleanup();
            return false;
        }
    }
    
    // 设置模型尺寸
    if (input_attrs[0].n_dims == 4) {
        if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
            model_channel = input_attrs[0].dims[1];
            model_height = input_attrs[0].dims[2];
            model_width = input_attrs[0].dims[3];
        } else {
            model_height = input_attrs[0].dims[1];
            model_width = input_attrs[0].dims[2];
            model_channel = input_attrs[0].dims[3];
        }
    } else {
        printf("❌ 错误: 不支持的输入维度数量: %d\n", input_attrs[0].n_dims);
        cleanup();
        return false;
    }
    
    // 检查是否为量化模型
    is_quant = (input_attrs[0].type == RKNN_TENSOR_INT8 || 
                input_attrs[0].type == RKNN_TENSOR_UINT8);
    
    is_initialized = true;
    
    return true;
}

void RknnContext::cleanup() {
    if (output_attrs) {
        delete[] output_attrs;
        output_attrs = nullptr;
    }
    
    if (input_attrs) {
        delete[] input_attrs;
        input_attrs = nullptr;
    }
    
    if (ctx) {
        rknn_destroy(ctx);
        ctx = 0;
    }
    
    is_initialized = false;
}

// ZeroCopyMemory 实现
ZeroCopyMemory::~ZeroCopyMemory() {
    cleanup();
}

bool ZeroCopyMemory::init(RknnContext* ctx) {
    if (!ctx || !ctx->is_initialized) {
        // RknnContext not initialized
        return false;
    }
    
    rknn_ctx = ctx;
    
    // 设置输入属性
    input_attr = ctx->input_attrs[0];
    input_attr.type = RKNN_TENSOR_UINT8;
    input_attr.fmt = RKNN_TENSOR_NHWC;
    
    // 创建输入内存
    input_mem = rknn_create_mem(ctx->ctx, input_attr.size_with_stride);
    if (!input_mem) {
        // Failed to create input memory
        return false;
    }
    
    // 绑定输入内存
    int ret = rknn_set_io_mem(ctx->ctx, input_mem, &input_attr);
    if (ret < 0) {
        // Failed to set input memory
        cleanup();
        return false;
    }
    
    // 创建输出内存
    output_mems.resize(ctx->io_num.n_output);
    output_attrs.resize(ctx->io_num.n_output);
    
    for (uint32_t i = 0; i < ctx->io_num.n_output; i++) {
        output_attrs[i] = ctx->output_attrs[i];
        output_mems[i] = rknn_create_mem(ctx->ctx, output_attrs[i].size_with_stride);
        if (!output_mems[i]) {
            // Failed to create output memory
            cleanup();
            return false;
        }
        
        ret = rknn_set_io_mem(ctx->ctx, output_mems[i], &output_attrs[i]);
        if (ret < 0) {
            // Failed to set output memory
            cleanup();
            return false;
        }
    }
    
    is_initialized = true;
    // ZeroCopyMemory initialized
    return true;
}

void ZeroCopyMemory::cleanup() {
    if (rknn_ctx && rknn_ctx->ctx) {
        for (auto mem : output_mems) {
            if (mem) {
                rknn_destroy_mem(rknn_ctx->ctx, mem);
            }
        }
        
        if (input_mem) {
            rknn_destroy_mem(rknn_ctx->ctx, input_mem);
        }
    }
    
    output_mems.clear();
    output_attrs.clear();
    input_mem = nullptr;
    rknn_ctx = nullptr;
    is_initialized = false;
}

} // namespace internal
} // namespace detector