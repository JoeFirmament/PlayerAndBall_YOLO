 /*
 * test_basketball_rknn.cpp
 * 用于测试 basketball_player_rk3588.rknn 输出结构和类别含义
 * 用法: ./test_basketball_rknn <model_path> <image_path>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#include "rknn_api.h"
#include <cmath>
#include <algorithm>

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("用法: %s <model_path> <image_path>\n", argv[0]);
        return -1;
    }
    const char* model_path = argv[1];
    const char* image_path = argv[2];

    // 1. 加载模型
    FILE* fp = fopen(model_path, "rb");
    if (!fp) {
        printf("模型文件打开失败: %s\n", model_path);
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    int model_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    void* model_data = malloc(model_size);
    fread(model_data, 1, model_size, fp);
    fclose(fp);

    rknn_context ctx;
    int ret = rknn_init(&ctx, model_data, model_size, 0, NULL);
    if (ret != 0) {
        printf("rknn_init 失败: %d\n", ret);
        free(model_data);
        return -1;
    }
    printf("模型加载成功\n");

    // 2. 查询输入输出信息
    rknn_input_output_num io_num;
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    rknn_tensor_attr input_attr;
    memset(&input_attr, 0, sizeof(input_attr));
    input_attr.index = 0;
    rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attr, sizeof(input_attr));
    printf("模型输入: index=%d, dims=[%d,%d,%d,%d], type=%d\n", input_attr.index, input_attr.dims[0], input_attr.dims[1], input_attr.dims[2], input_attr.dims[3], input_attr.type);
    printf("输入量化信息: qnt_type=%d, scale=%f, zero_point=%d\n", input_attr.qnt_type, input_attr.scale, input_attr.zp);

    // 3. 读取图片并预处理
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        printf("图片读取失败: %s\n", image_path);
        rknn_destroy(ctx);
        free(model_data);
        return -1;
    }
    int model_width = input_attr.dims[2];
    int model_height = input_attr.dims[1];
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(model_width, model_height));
    cv::Mat rgb_img;
    cv::cvtColor(resized_img, rgb_img, cv::COLOR_BGR2RGB);

    // 4. 根据输入类型进行相应的数据预处理
    rknn_input input;
    memset(&input, 0, sizeof(input));
    input.index = 0;
    input.pass_through = 0;
    
    cv::Mat processed_img; // 用于保存预处理后的图像数据
    
    if (input_attr.type == RKNN_TENSOR_FLOAT32) {
        printf("输入类型: FLOAT32，进行归一化处理\n");
        // float32 输入：转换为 [0,1] 范围
        rgb_img.convertTo(processed_img, CV_32F, 1.0/255.0);
        input.type = RKNN_TENSOR_FLOAT32;
        input.fmt = RKNN_TENSOR_NHWC;
        input.size = model_width * model_height * 3 * sizeof(float);
        input.buf = processed_img.data;
    } else if (input_attr.type == RKNN_TENSOR_UINT8) {
        printf("输入类型: UINT8，直接使用原始数据\n");
        // uint8 输入：直接使用 [0,255] 范围
        processed_img = rgb_img; // 直接使用
        input.type = RKNN_TENSOR_UINT8;
        input.fmt = RKNN_TENSOR_NHWC;
        input.size = model_width * model_height * 3;
        input.buf = processed_img.data;
    } else if (input_attr.type == RKNN_TENSOR_INT8) {
        printf("输入类型: INT8，使用RKNN转换时的预处理参数\n");
        // 根据RKNN转换参数: mean=[0,0,0], std=[255,255,255]
        // 预处理公式: (pixel - mean) / std = (pixel - 0) / 255
        // 然后量化到INT8: quantized = (normalized_value / scale) + zero_point
        printf("量化参数: scale=%.6f, zero_point=%d\n", input_attr.scale, input_attr.zp);
        
        processed_img = cv::Mat(model_height, model_width, CV_8SC3);
        for (int i = 0; i < model_height * model_width * 3; ++i) {
            // 步骤1: 归一化 (pixel - mean) / std = (pixel - 0) / 255
            float normalized = rgb_img.data[i] / 255.0f;
            
            // 步骤2: 量化到INT8
            // 公式: int8_val = (normalized_value / scale) + zero_point
            int quantized = (int)((normalized / input_attr.scale) + input_attr.zp);
            
            // 步骤3: 限制到 [-128, 127] 范围
            quantized = std::max(-128, std::min(127, quantized));
            processed_img.data[i] = (int8_t)quantized;
        }
        
        input.type = RKNN_TENSOR_INT8;
        input.fmt = RKNN_TENSOR_NHWC;
        input.size = model_width * model_height * 3;
        input.buf = processed_img.data;
    } else {
        printf("未支持的输入类型: %d\n", input_attr.type);
        rknn_destroy(ctx);
        free(model_data);
        return -1;
    }

    // 5. 设置输入
    ret = rknn_inputs_set(ctx, 1, &input);
    if (ret != 0) {
        printf("rknn_inputs_set 失败: %d\n", ret);
        rknn_destroy(ctx);
        free(model_data);
        return -1;
    }

    // 5. 推理
    ret = rknn_run(ctx, NULL);
    if (ret != 0) {
        printf("rknn_run 失败: %d\n", ret);
        rknn_destroy(ctx);
        free(model_data);
        return -1;
    }

    // 7. 查询输出张量数量，兼容多输出模型
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    printf("模型输出张量数量: %d\n", io_num.n_output);

    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; ++i) {
        outputs[i].want_float = 1;
        outputs[i].index = i;
    }
    int ret_outputs = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
    if (ret_outputs != 0) {
        printf("rknn_outputs_get 失败: %d\n", ret_outputs);
        rknn_destroy(ctx);
        free(model_data);
        return -1;
    }

    for (int out_idx = 0; out_idx < io_num.n_output; ++out_idx) {
        rknn_tensor_attr output_attr;
        memset(&output_attr, 0, sizeof(output_attr));
        output_attr.index = out_idx;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attr, sizeof(output_attr));
        printf("\n[输出张量 %d] dims=[%d,%d,%d,%d], n_dims=%d, type=%d\n", out_idx, output_attr.dims[0], output_attr.dims[1], output_attr.dims[2], output_attr.dims[3], output_attr.n_dims, output_attr.type);
        printf("量化类型: qnt_type=%d, scale=%f, zero_point=%d\n", output_attr.qnt_type, output_attr.scale, output_attr.zp);
        // 自动根据类型打印前10项（支持float32/int8/uint8）
        printf("前10项原始输出及反量化值：\n");
        if (output_attr.type == RKNN_TENSOR_FLOAT32) {
            float* out_ptr = (float*)outputs[out_idx].buf;
            for (int i = 0; i < 10; ++i) {
                printf("[%d] %.6f\n", i, out_ptr[i]);
            }
        } else if (output_attr.type == RKNN_TENSOR_INT8) {
            int8_t* out_ptr = (int8_t*)outputs[out_idx].buf;
            for (int i = 0; i < 10; ++i) {
                float v = (out_ptr[i] - output_attr.zp) * output_attr.scale;
                printf("[%d] raw=%d, dequant=%.6f\n", i, out_ptr[i], v);
            }
        } else if (output_attr.type == RKNN_TENSOR_UINT8) {
            uint8_t* out_ptr = (uint8_t*)outputs[out_idx].buf;
            for (int i = 0; i < 10; ++i) {
                float v = (out_ptr[i] - output_attr.zp) * output_attr.scale;
                printf("[%d] raw=%u, dequant=%.6f\n", i, out_ptr[i], v);
            }
        } else {
            printf("[警告] 未知输出类型: %d\n", output_attr.type);
        }
        // 如果输出为检测类（如6通道），打印前10组(x,y,w,h,conf,class)
        int num_attrs = 6;
        int n = 0;
        if (output_attr.n_dims >= 2 && (output_attr.dims[1] == 6 || output_attr.dims[0] == 6)) {
            int anchor_dim = (output_attr.dims[1] == 6) ? 2 : 1;
            int num_boxes = output_attr.dims[anchor_dim == 1 ? 1 : 2];
            
            // 分析原始输出分布
            printf("=== 原始输出分析 ===\n");
            if (output_attr.type == RKNN_TENSOR_INT8) {
                int8_t* raw_ptr = (int8_t*)outputs[out_idx].buf;
                for (int dim = 0; dim < 6; ++dim) {
                    printf("维度[%d]前20项: ", dim);
                    for (int i = 0; i < 20 && i < num_boxes; ++i) {
                        int idx = (anchor_dim == 1) ? (i * 6 + dim) : (dim * num_boxes + i);
                        float val = (raw_ptr[idx] - output_attr.zp) * output_attr.scale;
                        printf("%.2f ", val);
                    }
                    printf("\n");
                }
            }
            
            float* out_ptr = (float*)outputs[out_idx].buf;
            
            // === 尝试方式1: [batch, attr, anchor] 解析 ===
            printf("=== 方式1: [batch, attr, anchor] 解析 ===\n");
            printf("前10组输出数据 (x,y,w,h,conf,class):\n");
            for (int i = 0; i < 10 && i < num_boxes; ++i) {
                float x, y, w, h, conf, class_val;
                // 对于 shape [1,6,8400]，按 [batch, attr, anchor] 方式读取
                x = out_ptr[0*num_boxes + i];     // attr=0: x坐标
                y = out_ptr[1*num_boxes + i];     // attr=1: y坐标
                w = out_ptr[2*num_boxes + i];     // attr=2: 宽度
                h = out_ptr[3*num_boxes + i];     // attr=3: 高度
                conf = out_ptr[4*num_boxes + i];  // attr=4: 置信度
                class_val = out_ptr[5*num_boxes + i];  // attr=5: 类别
                
                // 尝试sigmoid激活
                float sigmoid_conf = 1.0f / (1.0f + exp(-conf));
                float sigmoid_class = 1.0f / (1.0f + exp(-class_val));
                printf("[%d] x=%.3f y=%.3f w=%.3f h=%.3f conf=%.3f(sigmoid:%.3f) class=%.3f(sigmoid:%.3f)\n", 
                       i, x, y, w, h, conf, sigmoid_conf, class_val, sigmoid_class);
            }
            
            // === 尝试方式2: [batch, anchor, attr] 解析 ===
            printf("\n=== 方式2: [batch, anchor, attr] 解析 ===\n");
            printf("前10组输出数据 (x,y,w,h,conf,class):\n");
            for (int i = 0; i < 10 && i < num_boxes; ++i) {
                float x, y, w, h, conf, class_val;
                // 按 [batch, anchor, attr] 方式读取
                x = out_ptr[i * 6 + 0];     // anchor i, attr 0
                y = out_ptr[i * 6 + 1];     // anchor i, attr 1
                w = out_ptr[i * 6 + 2];     // anchor i, attr 2
                h = out_ptr[i * 6 + 3];     // anchor i, attr 3
                conf = out_ptr[i * 6 + 4];  // anchor i, attr 4
                class_val = out_ptr[i * 6 + 5];  // anchor i, attr 5
                
                // 尝试sigmoid激活
                float sigmoid_conf = 1.0f / (1.0f + exp(-conf));
                float sigmoid_class = 1.0f / (1.0f + exp(-class_val));
                printf("[%d] x=%.3f y=%.3f w=%.3f h=%.3f conf=%.3f(sigmoid:%.3f) class=%.3f(sigmoid:%.3f)\n", 
                       i, x, y, w, h, conf, sigmoid_conf, class_val, sigmoid_class);
            }
            
            // === 查找高置信度的检测 ===
            printf("\n=== 查找高置信度检测 (两种解析方式) ===\n");
            
            // 先分析 conf 和 class 通道的数据范围
            printf("=== 数据范围分析 ===\n");
            
            // 方式1: [batch, attr, anchor]
            float min_conf1 = 999, max_conf1 = -999;
            float min_class1 = 999, max_class1 = -999;
            int zero_conf_count1 = 0, zero_class_count1 = 0;
            for (int i = 0; i < num_boxes; ++i) {
                float conf = out_ptr[4*num_boxes + i];
                float class_val = out_ptr[5*num_boxes + i];
                min_conf1 = std::min(min_conf1, conf);
                max_conf1 = std::max(max_conf1, conf);
                min_class1 = std::min(min_class1, class_val);
                max_class1 = std::max(max_class1, class_val);
                if (conf == 0.0f) zero_conf_count1++;
                if (class_val == 0.0f) zero_class_count1++;
            }
            printf("方式1 - conf范围: [%.3f, %.3f], 零值数量: %d/%d\n", 
                   min_conf1, max_conf1, zero_conf_count1, num_boxes);
            printf("方式1 - class范围: [%.3f, %.3f], 零值数量: %d/%d\n", 
                   min_class1, max_class1, zero_class_count1, num_boxes);
            
            // 方式2: [batch, anchor, attr]
            float min_conf2 = 999, max_conf2 = -999;
            float min_class2 = 999, max_class2 = -999;
            for (int i = 0; i < num_boxes; ++i) {
                float conf = out_ptr[i * 6 + 4];
                float class_val = out_ptr[i * 6 + 5];
                min_conf2 = std::min(min_conf2, conf);
                max_conf2 = std::max(max_conf2, conf);
                min_class2 = std::min(min_class2, class_val);
                max_class2 = std::max(max_class2, class_val);
            }
            printf("方式2 - conf范围: [%.3f, %.3f], class范围: [%.3f, %.3f]\n", 
                   min_conf2, max_conf2, min_class2, max_class2);
            
            // 检查是否模型输出全为零
            if (zero_conf_count1 == num_boxes) {
                printf("\n!!! 警告: 所有置信度都为0，可能的问题:\n");
                printf("1. 模型转换问题 - 检查RKNN转换过程\n");
                printf("2. 预处理不匹配 - 验证输入预处理方式\n");
                printf("3. 模型未正确训练 - 检查原始ONNX/PyTorch模型\n");
                printf("4. 输入图像问题 - 尝试其他测试图像\n");
                
                // 检查其他通道是否也全为零
                printf("\n=== 检查所有通道是否为零 ===\n");
                for (int ch = 0; ch < 6; ++ch) {
                    int zero_count = 0;
                    float min_val = 999, max_val = -999;
                    for (int i = 0; i < num_boxes; ++i) {
                        float val = out_ptr[ch * num_boxes + i];
                        if (val == 0.0f) zero_count++;
                        min_val = std::min(min_val, val);
                        max_val = std::max(max_val, val);
                    }
                    const char* ch_names[] = {"X", "Y", "W", "H", "CONF", "CLASS"};
                    printf("通道%d(%s): 零值%d/%d, 范围[%.3f, %.3f]\n", 
                           ch, ch_names[ch], zero_count, num_boxes, min_val, max_val);
                }
            }
        }
    }
    rknn_outputs_release(ctx, io_num.n_output, outputs);

    // 9. 释放资源
    rknn_destroy(ctx);
    free(model_data);
    printf("测试完成\n");
    return 0;
}
