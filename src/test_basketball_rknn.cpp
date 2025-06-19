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

    // 4. 设置输入
    rknn_input input;
    memset(&input, 0, sizeof(input));
    input.index = 0;
    input.type = RKNN_TENSOR_UINT8;
    input.size = model_width * model_height * 3;
    input.fmt = RKNN_TENSOR_NHWC;
    input.buf = rgb_img.data;
    input.pass_through = 0;
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
            float* out_ptr = (float*)outputs[out_idx].buf;
            printf("前10组输出数据 (x,y,w,h,conf,class):\n");
            for (int i = 0; i < 10 && i < num_boxes; ++i) {
                if (anchor_dim == 1) {
                    printf("[%d] x=%.3f y=%.3f w=%.3f h=%.3f conf=%.3f class=%.3f\n", i, out_ptr[i*6+0], out_ptr[i*6+1], out_ptr[i*6+2], out_ptr[i*6+3], out_ptr[i*6+4], out_ptr[i*6+5]);
                } else {
                    printf("[%d] x=%.3f y=%.3f w=%.3f h=%.3f conf=%.3f class=%.3f\n", i, out_ptr[0* num_boxes + i], out_ptr[1* num_boxes + i], out_ptr[2* num_boxes + i], out_ptr[3* num_boxes + i], out_ptr[4* num_boxes + i], out_ptr[5* num_boxes + i]);
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
