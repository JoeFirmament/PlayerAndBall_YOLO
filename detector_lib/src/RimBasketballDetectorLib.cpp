#include "RimBasketballDetectorLib.h"
#include "internal/detector_common.h"
#include <cmath>
#include <set>
#include <algorithm>

// 添加rim_basketball后处理需要的定义
#define MAX_DETECTIONS 100
#define BASKETBALL_CLASS_ID 0
#define RIM_CLASS_ID 1

// 检测结果结构体
typedef struct {
    float x, y, w, h;       // 边界框坐标 (center_x, center_y, width, height)
    float confidence;       // 置信度
    int class_id;           // 类别ID: 0=basketball, 1=rim
    const char* class_name; // 类别名称
} RimBasketballDetection;

typedef struct {
    RimBasketballDetection detections[MAX_DETECTIONS];
    int count;
} RimBasketballDetectionResult;

// 内部检测结果结构体
typedef struct {
    float xmin, ymin, xmax, ymax;
    float score;
    int class_id;
} DetectRect;

namespace detector {

// 类别名称
static const char* class_names[2] = {"basketball", "rim"};

// RimBasketballDetectorLib内部实现类
class RimBasketballDetectorLib::Impl {
public:
    explicit Impl(const std::string& model_path)
        : model_path_(model_path)
        , status_(DETECTOR_UNINITIALIZED)
        , conf_threshold_(0.25f)
        , nms_threshold_(0.1f)
        , last_inference_time_ms_(-1)
    {
    }
    
    ~Impl() {
        cleanup();
    }
    
    std::vector<RimBasketballResult> detect(const cv::Mat& frame) {
        if (frame.empty()) {
            return {};
        }
        
        // 延迟初始化
        if (!lazy_init()) {
            return {};
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // 1. 图像预处理到NPU内存 (letterbox resize)
        float scale;
        int x_pad, y_pad;
        if (!letterbox_resize_to_npu(frame, &scale, &x_pad, &y_pad)) {
            return {};
        }
        
        // 2. NPU推理
        int ret = rknn_run(rknn_ctx_.ctx, nullptr);
        if (ret < 0) {
            return {};
        }
        
        // 3. 后处理：调用真正的RKNN后处理函数
        std::vector<RimBasketballResult> results = postprocess_real_results(frame.size(), scale, x_pad, y_pad);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        last_inference_time_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        
        return results;
    }
    
    void set_confidence_threshold(float threshold) {
        conf_threshold_ = std::max(0.01f, std::min(0.99f, threshold));
    }
    
    void set_nms_threshold(float threshold) {
        nms_threshold_ = std::max(0.01f, std::min(0.99f, threshold));
    }
    
    bool is_initialized() const {
        return status_ == DETECTOR_READY;
    }
    
    DetectorStatus get_status() const {
        return status_;
    }
    
    void cleanup() {
        if (status_ != DETECTOR_UNINITIALIZED) {
            zero_copy_mem_.cleanup();
            rknn_ctx_.cleanup();
            status_ = DETECTOR_UNINITIALIZED;
        }
    }
    
    int get_last_inference_time_ms() const {
        return last_inference_time_ms_;
    }

private:
    // 配置参数
    std::string model_path_;
    DetectorStatus status_;
    float conf_threshold_;
    float nms_threshold_;
    int last_inference_time_ms_;
    
    // 内部资源
    internal::RknnContext rknn_ctx_;
    internal::ZeroCopyMemory zero_copy_mem_;
    
    bool lazy_init() {
        if (status_ == DETECTOR_READY) {
            return true;
        }
        
        if (status_ == DETECTOR_ERROR) {
            return false;
        }
        
        status_ = DETECTOR_INITIALIZING;
        
        // 1. 初始化RKNN模型
        if (!rknn_ctx_.init_from_file(model_path_)) {
            status_ = DETECTOR_ERROR;
            return false;
        }
        
        // 2. 初始化零拷贝内存
        if (!zero_copy_mem_.init(&rknn_ctx_)) {
            status_ = DETECTOR_ERROR;
            return false;
        }
        
        status_ = DETECTOR_READY;
        return true;
    }
    
    bool letterbox_resize_to_npu(const cv::Mat& src, float* scale, int* x_pad, int* y_pad) {
        if (!zero_copy_mem_.is_initialized) {
            return false;
        }
        
        int src_w = src.cols;
        int src_h = src.rows;
        int dst_w = rknn_ctx_.model_width;   // 640
        int dst_h = rknn_ctx_.model_height;  // 640
        
        // 计算缩放比例 (保持宽高比)
        *scale = std::min((float)dst_w / src_w, (float)dst_h / src_h);
        int new_w = (int)(src_w * (*scale));
        int new_h = (int)(src_h * (*scale));
        
        *x_pad = (dst_w - new_w) / 2;
        *y_pad = (dst_h - new_h) / 2;
        
        // 创建指向NPU内存的Mat
        cv::Mat npu_mat(dst_h, dst_w, CV_8UC3, zero_copy_mem_.input_mem->virt_addr);
        npu_mat.setTo(cv::Scalar(114, 114, 114)); // 灰色填充 (RGB顺序)
        
        // 关键修复：BGR转RGB
        cv::Mat src_rgb;
        cv::cvtColor(src, src_rgb, cv::COLOR_BGR2RGB);
        
        // resize原图到目标尺寸
        cv::Mat resized;
        cv::resize(src_rgb, resized, cv::Size(new_w, new_h));
        
        // 拷贝到NPU内存的中心位置
        cv::Rect roi(*x_pad, *y_pad, new_w, new_h);
        resized.copyTo(npu_mat(roi));
        
        return true;
    }
    
    // 真正的rim_basketball后处理函数 - 严格按照工作版本实现
    std::vector<RimBasketballResult> postprocess_real_results(const cv::Size& frame_size, float scale, int x_pad, int y_pad) {
        std::vector<RimBasketballResult> results;
        
        // 调用真正的rim_basketball后处理函数
        RimBasketballDetectionResult detection_result;
        int ret = real_rim_basketball_post_process(frame_size, scale, x_pad, y_pad, conf_threshold_, nms_threshold_, &detection_result);
        
        if (ret != 0) {
            printf("❌ 篮筐篮球后处理失败! ret=%d\n", ret);
            return results;
        }
        
        // 转换为我们的RimBasketballResult格式
        for (int i = 0; i < detection_result.count; i++) {
            const RimBasketballDetection* det = &detection_result.detections[i];
            
            RimBasketballResult result;
            result.class_id = det->class_id;
            result.class_name = (det->class_id == 0) ? "basketball" : "rim";
            result.confidence = det->confidence;
            
            // 坐标映射：将模型输出坐标转换回原始图像坐标
            float x1 = det->x - det->w / 2.0f;
            float y1 = det->y - det->h / 2.0f;
            float x2 = det->x + det->w / 2.0f;
            float y2 = det->y + det->h / 2.0f;
            
            // 逆letterbox映射：640x640 → 原始图像尺寸
            x1 = (x1 - x_pad) / scale;
            y1 = (y1 - y_pad) / scale;
            x2 = (x2 - x_pad) / scale;
            y2 = (y2 - y_pad) / scale;
            
            // 限制到图像边界内
            x1 = std::max(0.0f, std::min(x1, (float)(frame_size.width - 1)));
            y1 = std::max(0.0f, std::min(y1, (float)(frame_size.height - 1)));
            x2 = std::max(0.0f, std::min(x2, (float)(frame_size.width - 1)));
            y2 = std::max(0.0f, std::min(y2, (float)(frame_size.height - 1)));
            
            // 设置边界框和中心点
            result.bbox = cv::Rect((int)x1, (int)y1, (int)(x2-x1), (int)(y2-y1));
            result.center = cv::Point2f((x1 + x2) / 2.0f, (y1 + y2) / 2.0f);
            result.distance_to_rim = 0.0f;
            result.is_in_rim_roi = false;
            
            results.push_back(result);
        }
        
        // 计算篮球到篮筐的距离和ROI分析
        analyze_basketball_rim_distance(results);
        
        return results;
    }
    
    // 严格按照工作版本实现的真正后处理函数  
    int real_rim_basketball_post_process(const cv::Size& frame_size, float scale, int x_pad, int y_pad,
                                        float conf_threshold, float nms_threshold, 
                                        RimBasketballDetectionResult* result) {
        result->count = 0;
        
        // 获取RKNN输出 - 完全按照工作版本方式
        rknn_output outputs[rknn_ctx_.io_num.n_output];
        memset(outputs, 0, sizeof(outputs));
        for (int i = 0; i < rknn_ctx_.io_num.n_output; i++) {
            outputs[i].index = i;
            outputs[i].want_float = (!rknn_ctx_.is_quant);
        }
        int ret = rknn_outputs_get(rknn_ctx_.ctx, rknn_ctx_.io_num.n_output, outputs, NULL);
        if (ret < 0) {
            printf("获取输出失败! ret=%d\n", ret);
            return ret;
        }
        
        // 模型参数
        const int strides[3] = {8, 16, 32};
        const int map_sizes[3][2] = {{80, 80}, {40, 40}, {20, 20}};
        
        // 获取输出指针和量化参数 - 6输出格式: reg1, cls1, reg2, cls2, reg3, cls3
        int8_t* reg_outputs[3] = {(int8_t*)outputs[0].buf, (int8_t*)outputs[2].buf, (int8_t*)outputs[4].buf};
        int8_t* cls_outputs[3] = {(int8_t*)outputs[1].buf, (int8_t*)outputs[3].buf, (int8_t*)outputs[5].buf};
        
        std::vector<DetectRect> detect_rects;
        int basketball_count = 0;
        int rim_count = 0;
        
        // 处理3个检测层
        for (int layer = 0; layer < 3; layer++) {
            int stride = strides[layer];
            int height = map_sizes[layer][0];
            int width = map_sizes[layer][1];
            
            int8_t* reg_data = reg_outputs[layer];
            int8_t* cls_data = cls_outputs[layer];
            
            // 量化参数
            int reg_zp = rknn_ctx_.output_attrs[layer * 2].zp;
            float reg_scale = rknn_ctx_.output_attrs[layer * 2].scale;
            int cls_zp = rknn_ctx_.output_attrs[layer * 2 + 1].zp;
            float cls_scale = rknn_ctx_.output_attrs[layer * 2 + 1].scale;
            
            
            // 遍历网格
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    // 获取最高置信度的类别
                    float max_conf = 0.0f;
                    int best_class = 0;
                    
                    // 使用量化阈值进行快速筛选
                    int8_t thres_i8 = qnt_f32_to_affine(unsigmoid(conf_threshold), cls_zp, cls_scale);
                    
                    for (int c = 0; c < 2; c++) { // 2个类别：篮球(0), 篮筐(1)
                        int cls_idx = c * height * width + h * width + w;
                        
                        if (cls_data[cls_idx] >= thres_i8) {
                            float raw_score = deqnt_affine_to_f32(cls_data[cls_idx], cls_zp, cls_scale);
                            float conf = sigmoid(raw_score);
                            
                            if (conf > max_conf) {
                                max_conf = conf;
                                best_class = c;
                            }
                        }
                    }
                    
                    // 检查是否超过阈值
                    if (max_conf > conf_threshold) {
                        int grid_pos = h * width + w;
                        int hw_size = height * width;
                        
                        if (grid_pos >= hw_size) continue;
                        
                        // 按照[4, H*W]格式访问
                        float left_dist   = deqnt_affine_to_f32(reg_data[0 * hw_size + grid_pos], reg_zp, reg_scale);
                        float top_dist    = deqnt_affine_to_f32(reg_data[1 * hw_size + grid_pos], reg_zp, reg_scale);
                        float right_dist  = deqnt_affine_to_f32(reg_data[2 * hw_size + grid_pos], reg_zp, reg_scale);
                        float bottom_dist = deqnt_affine_to_f32(reg_data[3 * hw_size + grid_pos], reg_zp, reg_scale);
                        
                        // 计算anchor center
                        float anchor_x = (w + 0.5f) * stride;
                        float anchor_y = (h + 0.5f) * stride;
                        
                        // 基于DFL处理后的距离计算边界框
                        float x1 = anchor_x - left_dist * stride;
                        float y1 = anchor_y - top_dist * stride;
                        float x2 = anchor_x + right_dist * stride;
                        float y2 = anchor_y + bottom_dist * stride;
                        
                        // 边界检查
                        x1 = clamp(x1, 0.0f, 640.0f);
                        y1 = clamp(y1, 0.0f, 640.0f);
                        x2 = clamp(x2, 0.0f, 640.0f);
                        y2 = clamp(y2, 0.0f, 640.0f);
                        
                        if (x1 < x2 && y1 < y2) {
                            DetectRect rect;
                            rect.xmin = x1 / 640.0f;
                            rect.ymin = y1 / 640.0f;
                            rect.xmax = x2 / 640.0f;
                            rect.ymax = y2 / 640.0f;
                            rect.score = max_conf;
                            rect.class_id = best_class;
                            
                            detect_rects.push_back(rect);
                            if (best_class == 0) basketball_count++;
                            else if (best_class == 1) rim_count++;
                        }
                    }
                }
            }
        }
        
        if (detect_rects.empty()) {
            rknn_outputs_release(rknn_ctx_.ctx, rknn_ctx_.io_num.n_output, outputs);
            return 0;
        }
        
        // 按置信度排序
        std::sort(detect_rects.begin(), detect_rects.end(),
                  [](const DetectRect& a, const DetectRect& b) {
                      return a.score > b.score;
                  });
        
        // NMS处理
        std::vector<bool> suppressed(detect_rects.size(), false);
        
        for (int i = 0; i < detect_rects.size(); i++) {
            if (suppressed[i]) continue;
            
            const DetectRect& rect_i = detect_rects[i];
            
            for (int j = i + 1; j < detect_rects.size(); j++) {
                if (suppressed[j]) continue;
                
                const DetectRect& rect_j = detect_rects[j];
                
                // 同类别才进行NMS
                if (rect_i.class_id == rect_j.class_id) {
                    float iou = calculate_iou(rect_i.xmin, rect_i.ymin, rect_i.xmax, rect_i.ymax,
                                            rect_j.xmin, rect_j.ymin, rect_j.xmax, rect_j.ymax);
                    
                    if (iou > nms_threshold) {
                        suppressed[j] = true;
                    }
                }
            }
        }
        
        // 构建最终结果
        for (int i = 0; i < detect_rects.size() && result->count < MAX_DETECTIONS; i++) {
            if (suppressed[i]) continue;
            
            const DetectRect& rect = detect_rects[i];
            RimBasketballDetection* det = &result->detections[result->count];
            
            // 转换回像素坐标
            det->x = (rect.xmin + rect.xmax) / 2.0f * 640;  // center_x
            det->y = (rect.ymin + rect.ymax) / 2.0f * 640;  // center_y
            det->w = (rect.xmax - rect.xmin) * 640;         // width
            det->h = (rect.ymax - rect.ymin) * 640;         // height
            det->confidence = rect.score;
            det->class_id = rect.class_id;
            det->class_name = class_names[rect.class_id];
            
            result->count++;
        }
        
        rknn_outputs_release(rknn_ctx_.ctx, rknn_ctx_.io_num.n_output, outputs);
        
        return 0;
    }
    
    void analyze_basketball_rim_distance(std::vector<RimBasketballResult>& results) {
        cv::Point2f rim_center(-1, -1);
        bool has_rim = false;
        
        // 找到篮筐位置
        for (const auto& result : results) {
            if (result.class_id == 1) { // rim
                rim_center = result.center;
                has_rim = true;
                break;
            }
        }
        
        // 为每个篮球计算距离
        for (auto& result : results) {
            if (result.class_id == 0) { // basketball
                if (has_rim) {
                    float dx = result.center.x - rim_center.x;
                    float dy = result.center.y - rim_center.y;
                    result.distance_to_rim = std::sqrt(dx * dx + dy * dy);
                    result.is_in_rim_roi = (result.distance_to_rim < 120.0f);
                } else {
                    result.distance_to_rim = -1.0f;
                    result.is_in_rim_roi = false;
                }
            }
        }
    }
    
    // 真正的反量化函数
    float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) {
        return ((float)qnt - (float)zp) * scale;
    }
    
    int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale) {
        float dst_val = (f32 / scale) + zp;
        return (int8_t)std::max(-128.0f, std::min(127.0f, dst_val));
    }
    
    float unsigmoid(float y) {
        return -logf((1.0f / y) - 1.0f);
    }
    
    float sigmoid(float x) {
        return 1.0f / (1.0f + expf(-x));
    }
    
    float clamp(float val, float min_val, float max_val) {
        return std::max(min_val, std::min(val, max_val));
    }
    
    // 计算IoU
    float calculate_iou(float xmin1, float ymin1, float xmax1, float ymax1,
                       float xmin2, float ymin2, float xmax2, float ymax2) {
        float xmin = std::max(xmin1, xmin2);
        float ymin = std::max(ymin1, ymin2);
        float xmax = std::min(xmax1, xmax2);
        float ymax = std::min(ymax1, ymax2);
        
        float inter_width = xmax - xmin;
        float inter_height = ymax - ymin;
        
        if (inter_width <= 0 || inter_height <= 0) return 0.0f;
        
        float intersection = inter_width * inter_height;
        float area1 = (xmax1 - xmin1) * (ymax1 - ymin1);
        float area2 = (xmax2 - xmin2) * (ymax2 - ymin2);
        float union_area = area1 + area2 - intersection;
        
        return union_area > 0 ? intersection / union_area : 0.0f;
    }
};

// RimBasketballDetectorLib公共接口实现
RimBasketballDetectorLib::RimBasketballDetectorLib(const std::string& model_path)
    : pImpl_(std::make_unique<Impl>(model_path)) {
}

RimBasketballDetectorLib::~RimBasketballDetectorLib() = default;

std::vector<RimBasketballResult> RimBasketballDetectorLib::detect(const cv::Mat& frame) {
    return pImpl_->detect(frame);
}

void RimBasketballDetectorLib::set_confidence_threshold(float threshold) {
    pImpl_->set_confidence_threshold(threshold);
}

void RimBasketballDetectorLib::set_nms_threshold(float threshold) {
    pImpl_->set_nms_threshold(threshold);
}

bool RimBasketballDetectorLib::is_initialized() const {
    return pImpl_->is_initialized();
}

DetectorStatus RimBasketballDetectorLib::get_status() const {
    return pImpl_->get_status();
}

void RimBasketballDetectorLib::release() {
    pImpl_->cleanup();
}

int RimBasketballDetectorLib::get_last_inference_time_ms() const {
    return pImpl_->get_last_inference_time_ms();
}

std::vector<std::string> RimBasketballDetectorLib::get_supported_classes() {
    return {"basketball", "rim"};
}

} // namespace detector