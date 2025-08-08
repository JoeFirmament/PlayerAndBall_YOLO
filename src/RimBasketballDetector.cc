#include "RimBasketballDetector.h"
#include "rim_basketball_postprocess.h"
#include "rknn_api.h"
#include "common.h"
#include "im2d.h"
#include "RgaUtils.h"
#include <cstring>
#include <algorithm>

// RimBasketballDetector的内部实现类
class RimBasketballDetector::Impl {
public:
    explicit Impl(const std::string& model_path)
        : model_path_(model_path)
        , initialized_(false)
        , conf_threshold_(0.25f)
        , nms_threshold_(0.45f)
        , input_mem_(nullptr)
    {
        memset(&app_ctx_, 0, sizeof(app_ctx_));
        memset(&zero_copy_ctx_, 0, sizeof(zero_copy_ctx_));
        memset(class_names_, 0, sizeof(class_names_));
        
        // 设置类别名称
        class_names_[0] = "basketball";
        class_names_[1] = "rim";
    }
    
    ~Impl() {
        cleanup_all_resources();
    }
    
    bool lazy_initialize() {
        if (initialized_) return true;
        
        printf("正在初始化RimBasketballDetector: %s\n", model_path_.c_str());
        
        // 按顺序初始化所有资源
        if (!init_rknn_model()) {
            printf("错误: RKNN模型初始化失败\n");
            return false;
        }
        
        if (!init_zero_copy_memory()) {
            printf("错误: 零拷贝内存初始化失败\n");
            cleanup_all_resources();
            return false;
        }
        
        initialized_ = true;
        printf("RimBasketballDetector初始化成功\n");
        return true;
    }
    
    std::vector<RimBasketballResult> detect(const cv::Mat& frame) {
        // 延迟初始化
        if (!lazy_initialize()) {
            return {};
        }
        
        if (frame.empty()) {
            printf("警告: 输入图像为空\n");
            return {};
        }
        
        // 1. 图像预处理到NPU内存
        if (!preprocess_frame_to_npu_memory(frame)) {
            printf("错误: 图像预处理失败\n");
            return {};
        }
        
        // 2. NPU推理
        int ret = rknn_run(app_ctx_.rknn_ctx, nullptr);
        if (ret < 0) {
            printf("错误: RKNN推理失败, ret=%d\n", ret);
            return {};
        }
        
        // 3. 后处理
        RimBasketballDetectionResult detect_results;
        memset(&detect_results, 0, sizeof(detect_results));
        
        // 准备输出数据
        rknn_output outputs[app_ctx_.io_num.n_output];
        memset(outputs, 0, sizeof(outputs));
        for (int i = 0; i < app_ctx_.io_num.n_output; i++) {
            outputs[i].want_float = 0;
            outputs[i].is_prealloc = 1;
            outputs[i].buf = zero_copy_ctx_.output_mems[i]->virt_addr;
            outputs[i].size = zero_copy_ctx_.output_attrs[i].size_with_stride;
        }
        
        ret = process_rim_basketball_outputs(outputs, zero_copy_ctx_.output_attrs, 
                                           conf_threshold_, nms_threshold_, &detect_results);
        if (ret != 0) {
            printf("错误: 后处理失败\n");
            return {};
        }
        
        // 4. 转换为RimBasketballResult格式
        std::vector<RimBasketballResult> results;
        convert_detection_results(detect_results, results, frame.size());
        
        // 5. 分析篮球与篮筐的关系
        analyze_basketball_rim_relationship(results);
        
        return results;
    }
    
    void set_confidence_threshold(float threshold) {
        conf_threshold_ = std::max(0.01f, std::min(0.99f, threshold));
    }
    
    void set_nms_threshold(float threshold) {
        nms_threshold_ = std::max(0.01f, std::min(0.99f, threshold));
    }
    
    bool is_initialized() const {
        return initialized_;
    }
    
    void cleanup_all_resources() {
        if (!initialized_) return;
        
        cleanup_zero_copy_memory();
        cleanup_rknn_model();
        
        initialized_ = false;
    }

private:
    // 配置参数
    std::string model_path_;
    bool initialized_;
    float conf_threshold_;
    float nms_threshold_;
    const char* class_names_[2];
    
    // RKNN相关
    typedef struct {
        rknn_context rknn_ctx;
        rknn_input_output_num io_num;
        rknn_tensor_attr* input_attrs;
        rknn_tensor_attr* output_attrs;
        int model_channel;
        int model_width;
        int model_height;
        bool is_quant;
    } rknn_app_context_t;
    rknn_app_context_t app_ctx_;
    
    // 零拷贝内存管理
    rknn_tensor_mem* input_mem_;
    typedef struct {
        rknn_tensor_mem* input_mem;
        rknn_tensor_mem* output_mems[10];
        rknn_tensor_attr input_attr;
        rknn_tensor_attr output_attrs[10];
        int model_width;
        int model_height;
        int model_channels;
    } rim_zero_copy_context_t;
    rim_zero_copy_context_t zero_copy_ctx_;
    
    bool init_rknn_model() {
        // 读取模型文件
        FILE* fp = fopen(model_path_.c_str(), "rb");
        if (!fp) {
            printf("错误: 无法打开模型文件: %s\n", model_path_.c_str());
            return false;
        }
        
        fseek(fp, 0, SEEK_END);
        int model_size = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        
        void* model_data = malloc(model_size);
        if (!model_data) {
            fclose(fp);
            printf("错误: 内存分配失败\n");
            return false;
        }
        
        fread(model_data, 1, model_size, fp);
        fclose(fp);
        
        // 初始化RKNN上下文
        int ret = rknn_init(&app_ctx_.rknn_ctx, model_data, model_size, 0, NULL);
        free(model_data);
        
        if (ret < 0) {
            printf("错误: RKNN初始化失败, ret=%d\n", ret);
            return false;
        }
        
        // 获取模型输入输出信息
        ret = rknn_query(app_ctx_.rknn_ctx, RKNN_QUERY_IN_OUT_NUM, &app_ctx_.io_num, sizeof(app_ctx_.io_num));
        if (ret < 0) {
            printf("错误: 查询输入输出数量失败, ret=%d\n", ret);
            return false;
        }
        
        // 分配并获取输入属性
        app_ctx_.input_attrs = (rknn_tensor_attr*)malloc(app_ctx_.io_num.n_input * sizeof(rknn_tensor_attr));
        memset(app_ctx_.input_attrs, 0, app_ctx_.io_num.n_input * sizeof(rknn_tensor_attr));
        for (int i = 0; i < app_ctx_.io_num.n_input; i++) {
            app_ctx_.input_attrs[i].index = i;
            ret = rknn_query(app_ctx_.rknn_ctx, RKNN_QUERY_INPUT_ATTR, &(app_ctx_.input_attrs[i]), sizeof(rknn_tensor_attr));
            if (ret < 0) {
                printf("错误: 查询输入属性失败, ret=%d\n", ret);
                return false;
            }
        }
        
        // 分配并获取输出属性
        app_ctx_.output_attrs = (rknn_tensor_attr*)malloc(app_ctx_.io_num.n_output * sizeof(rknn_tensor_attr));
        memset(app_ctx_.output_attrs, 0, app_ctx_.io_num.n_output * sizeof(rknn_tensor_attr));
        for (int i = 0; i < app_ctx_.io_num.n_output; i++) {
            app_ctx_.output_attrs[i].index = i;
            ret = rknn_query(app_ctx_.rknn_ctx, RKNN_QUERY_OUTPUT_ATTR, &(app_ctx_.output_attrs[i]), sizeof(rknn_tensor_attr));
            if (ret < 0) {
                printf("错误: 查询输出属性失败, ret=%d\n", ret);
                return false;
            }
        }
        
        // 设置模型参数
        if (app_ctx_.input_attrs[0].n_dims == 4) {
            app_ctx_.model_channel = app_ctx_.input_attrs[0].dims[1];
            app_ctx_.model_height = app_ctx_.input_attrs[0].dims[2];
            app_ctx_.model_width = app_ctx_.input_attrs[0].dims[3];
        } else {
            printf("错误: 不支持的输入维度\n");
            return false;
        }
        
        printf("模型信息: %dx%dx%d\n", app_ctx_.model_width, app_ctx_.model_height, app_ctx_.model_channel);
        return true;
    }
    
    bool init_zero_copy_memory() {
        // 设置输入属性
        zero_copy_ctx_.input_attr = app_ctx_.input_attrs[0];
        zero_copy_ctx_.input_attr.type = RKNN_TENSOR_UINT8;
        zero_copy_ctx_.input_attr.fmt = RKNN_TENSOR_NHWC;
        zero_copy_ctx_.model_width = app_ctx_.model_width;
        zero_copy_ctx_.model_height = app_ctx_.model_height;
        zero_copy_ctx_.model_channels = app_ctx_.model_channel;
        
        // 创建输入内存
        zero_copy_ctx_.input_mem = rknn_create_mem(app_ctx_.rknn_ctx, zero_copy_ctx_.input_attr.size_with_stride);
        if (!zero_copy_ctx_.input_mem) {
            printf("错误: 创建输入内存失败\n");
            return false;
        }
        
        // 绑定输入内存
        int ret = rknn_set_io_mem(app_ctx_.rknn_ctx, zero_copy_ctx_.input_mem, &zero_copy_ctx_.input_attr);
        if (ret < 0) {
            printf("错误: 绑定输入内存失败, ret=%d\n", ret);
            return false;
        }
        
        // 创建输出内存
        for (int i = 0; i < app_ctx_.io_num.n_output; i++) {
            zero_copy_ctx_.output_attrs[i] = app_ctx_.output_attrs[i];
            zero_copy_ctx_.output_mems[i] = rknn_create_mem(app_ctx_.rknn_ctx, zero_copy_ctx_.output_attrs[i].size_with_stride);
            if (!zero_copy_ctx_.output_mems[i]) {
                printf("错误: 创建输出内存 %d 失败\n", i);
                return false;
            }
            ret = rknn_set_io_mem(app_ctx_.rknn_ctx, zero_copy_ctx_.output_mems[i], &zero_copy_ctx_.output_attrs[i]);
            if (ret < 0) {
                printf("错误: 绑定输出内存 %d 失败, ret=%d\n", i, ret);
                return false;
            }
        }
        
        input_mem_ = zero_copy_ctx_.input_mem;
        return true;
    }
    
    bool preprocess_frame_to_npu_memory(const cv::Mat& frame) {
        // 使用RGA进行resize到NPU内存
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(zero_copy_ctx_.model_width, zero_copy_ctx_.model_height));
        
        // 确保格式为RGB
        cv::Mat rgb_frame;
        if (frame.channels() == 3) {
            cv::cvtColor(resized, rgb_frame, cv::COLOR_BGR2RGB);
        } else {
            rgb_frame = resized;
        }
        
        // 拷贝到NPU内存
        memcpy(zero_copy_ctx_.input_mem->virt_addr, rgb_frame.data, 
               zero_copy_ctx_.model_width * zero_copy_ctx_.model_height * zero_copy_ctx_.model_channels);
        
        return true;
    }
    
    void convert_detection_results(const RimBasketballDetectionResult& detect_results,
                                 std::vector<RimBasketballResult>& rim_results,
                                 const cv::Size& frame_size) {
        rim_results.clear();
        rim_results.reserve(detect_results.count);
        
        float scale_x = (float)frame_size.width / zero_copy_ctx_.model_width;
        float scale_y = (float)frame_size.height / zero_copy_ctx_.model_height;
        
        for (int i = 0; i < detect_results.count; i++) {
            const auto& det = detect_results.detections[i];
            
            RimBasketballResult result;
            result.class_id = det.class_id;
            result.class_name = class_names_[det.class_id];
            result.confidence = det.confidence;
            result.distance_to_rim = 0.0f;
            result.is_in_rim_roi = false;
            
            // 转换边界框坐标 (center_x, center_y, width, height -> x, y, w, h)
            int x = (int)((det.x - det.w / 2) * scale_x);
            int y = (int)((det.y - det.h / 2) * scale_y);
            int w = (int)(det.w * scale_x);
            int h = (int)(det.h * scale_y);
            
            result.bbox = cv::Rect(x, y, w, h);
            result.center = cv::Point2f(det.x * scale_x, det.y * scale_y);
            
            rim_results.push_back(result);
        }
    }
    
    void analyze_basketball_rim_relationship(std::vector<RimBasketballResult>& results) {
        // 找到所有篮筐和篮球
        std::vector<RimBasketballResult*> rims;
        std::vector<RimBasketballResult*> basketballs;
        
        for (auto& result : results) {
            if (result.class_id == 1) {  // rim
                rims.push_back(&result);
            } else if (result.class_id == 0) {  // basketball
                basketballs.push_back(&result);
            }
        }
        
        // 为每个篮球计算到最近篮筐的距离
        for (auto* basketball : basketballs) {
            float min_distance = std::numeric_limits<float>::max();
            RimBasketballResult* closest_rim = nullptr;
            
            for (auto* rim : rims) {
                float dx = basketball->center.x - rim->center.x;
                float dy = basketball->center.y - rim->center.y;
                float distance = std::sqrt(dx * dx + dy * dy);
                
                if (distance < min_distance) {
                    min_distance = distance;
                    closest_rim = rim;
                }
            }
            
            if (closest_rim) {
                basketball->distance_to_rim = min_distance;
                
                // 判断篮球是否在篮筐ROI区域内 (简单的距离阈值)
                float rim_roi_threshold = std::max(closest_rim->bbox.width, closest_rim->bbox.height) * 1.5f;
                basketball->is_in_rim_roi = (min_distance < rim_roi_threshold);
            }
        }
    }
    
    void cleanup_rknn_model() {
        if (app_ctx_.rknn_ctx) {
            rknn_destroy(app_ctx_.rknn_ctx);
            app_ctx_.rknn_ctx = 0;
        }
        
        if (app_ctx_.input_attrs) {
            free(app_ctx_.input_attrs);
            app_ctx_.input_attrs = nullptr;
        }
        
        if (app_ctx_.output_attrs) {
            free(app_ctx_.output_attrs);
            app_ctx_.output_attrs = nullptr;
        }
    }
    
    void cleanup_zero_copy_memory() {
        for (int i = 0; i < 10; i++) {
            if (zero_copy_ctx_.output_mems[i]) {
                rknn_destroy_mem(app_ctx_.rknn_ctx, zero_copy_ctx_.output_mems[i]);
                zero_copy_ctx_.output_mems[i] = nullptr;
            }
        }
        
        if (zero_copy_ctx_.input_mem) {
            rknn_destroy_mem(app_ctx_.rknn_ctx, zero_copy_ctx_.input_mem);
            zero_copy_ctx_.input_mem = nullptr;
        }
        
        input_mem_ = nullptr;
    }
};

// RimBasketballDetector公共接口实现
RimBasketballDetector::RimBasketballDetector(const std::string& model_path)
    : pImpl_(std::make_unique<Impl>(model_path)) {
}

RimBasketballDetector::~RimBasketballDetector() = default;

std::vector<RimBasketballResult> RimBasketballDetector::detect(const cv::Mat& frame) {
    return pImpl_->detect(frame);
}

void RimBasketballDetector::set_confidence_threshold(float threshold) {
    pImpl_->set_confidence_threshold(threshold);
}

void RimBasketballDetector::set_nms_threshold(float threshold) {
    pImpl_->set_nms_threshold(threshold);
}

bool RimBasketballDetector::is_initialized() const {
    return pImpl_->is_initialized();
}

void RimBasketballDetector::destroy() {
    pImpl_->cleanup_all_resources();
}