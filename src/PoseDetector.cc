#include "PoseDetector.h"
#include "pose_yolov8.h"
#include "pose_postprocess.h"
#include "pose_letterbox_utils.h"
#include "BYTETracker.h"
#include "image_utils.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <cstring>

// PoseDetector的内部实现类
class PoseDetector::Impl {
public:
    explicit Impl(const std::string& model_path)
        : model_path_(model_path)
        , initialized_(false)
        , tracking_enabled_(true)
        , conf_threshold_(0.25f)
        , has_homography_(false)
        , input_mem_(nullptr)
    {
        // 清空输出内存数组
        memset(output_mems_, 0, sizeof(output_mems_));
    }
    
    ~Impl() {
        cleanup_all_resources();
    }
    
    bool lazy_initialize() {
        if (initialized_) return true;
        
        printf("正在初始化PoseDetector: %s\n", model_path_.c_str());
        
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
        
        if (!init_letterbox_context()) {
            printf("错误: Letterbox上下文初始化失败\n");
            cleanup_all_resources();
            return false;
        }
        
        if (!init_byte_tracker()) {
            printf("错误: ByteTracker初始化失败\n");
            cleanup_all_resources();
            return false;
        }
        
        initialized_ = true;
        printf("PoseDetector初始化成功\n");
        return true;
    }
    
    std::vector<PoseResult> detect(const cv::Mat& frame) {
        // 延迟初始化
        if (!lazy_initialize()) {
            return {};
        }
        
        if (frame.empty()) {
            printf("警告: 输入图像为空\n");
            return {};
        }
        
        // 1. Letterbox预处理到NPU内存
        if (!letterbox_resize_to_npu_memory(frame)) {
            printf("错误: 图像预处理失败\n");
            return {};
        }
        
        // 2. NPU推理
        object_detect_result_list detect_results;
        memset(&detect_results, 0, sizeof(detect_results));
        
        int ret = process_yolov8_pose_outputs(&app_ctx_, &zero_copy_ctx_, &detect_results);
        if (ret != 0) {
            printf("错误: 姿态检测推理失败\n");
            return {};
        }
        
        // 3. 转换为PoseResult格式
        std::vector<PoseResult> results;
        convert_detection_results(detect_results, results, frame.size());
        
        // 4. 可选的ByteTrack跟踪
        if (tracking_enabled_ && byte_tracker_) {
            apply_byte_tracking(results);
        }
        
        // 5. 可选的Homography坐标映射
        if (has_homography_) {
            apply_homography_mapping(results);
        }
        
        return results;
    }
    
    void enable_tracking(bool enable) {
        tracking_enabled_ = enable;
        if (enable && !byte_tracker_) {
            init_byte_tracker();
        }
    }
    
    bool load_calibration(const std::string& calibration_file) {
        // 简化版本：暂时跳过JSON解析，返回false表示未启用Homography
        printf("警告: 标定文件加载功能需要依赖JSON库，当前版本暂未启用\n");
        printf("       如需启用，请安装libjsoncpp-dev并重新编译\n");
        return false;
    }
    
    void set_confidence_threshold(float threshold) {
        conf_threshold_ = std::max(0.01f, std::min(0.99f, threshold));
    }
    
    bool is_initialized() const {
        return initialized_;
    }
    
    void cleanup_all_resources() {
        if (!initialized_) return;
        
        // 按相反顺序清理资源
        cleanup_byte_tracker();
        cleanup_letterbox();
        cleanup_zero_copy_memory();
        cleanup_rknn_model();
        
        initialized_ = false;
    }

private:
    // 配置参数
    std::string model_path_;
    bool initialized_;
    bool tracking_enabled_;
    float conf_threshold_;
    bool has_homography_;
    cv::Mat homography_matrix_;
    
    // RKNN相关
    rknn_app_context_t app_ctx_;
    
    // 零拷贝内存管理
    rknn_tensor_mem* input_mem_;
    rknn_tensor_mem* output_mems_[4];
    typedef struct {
        rknn_tensor_mem* input_mem;
        rknn_tensor_mem* output_mems[4];
        rknn_tensor_attr input_attr;
        rknn_tensor_attr output_attrs[4];
        int model_width;
        int model_height;
        int model_channels;
        letterbox_context_t letterbox_ctx;
    } zero_copy_context_t;
    zero_copy_context_t zero_copy_ctx_;
    
    // ByteTracker
    std::unique_ptr<BYTETracker> byte_tracker_;
    
    bool init_rknn_model() {
        memset(&app_ctx_, 0, sizeof(app_ctx_));
        int ret = init_yolov8_pose_model_with_npu(model_path_.c_str(), &app_ctx_, 1);
        return ret == 0;
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
            return false;
        }
        
        // 绑定输入内存
        int ret = rknn_set_io_mem(app_ctx_.rknn_ctx, zero_copy_ctx_.input_mem, &zero_copy_ctx_.input_attr);
        if (ret < 0) {
            return false;
        }
        
        // 创建输出内存
        for (int i = 0; i < app_ctx_.io_num.n_output; i++) {
            zero_copy_ctx_.output_attrs[i] = app_ctx_.output_attrs[i];
            zero_copy_ctx_.output_mems[i] = rknn_create_mem(app_ctx_.rknn_ctx, zero_copy_ctx_.output_attrs[i].size_with_stride);
            if (!zero_copy_ctx_.output_mems[i]) {
                return false;
            }
            ret = rknn_set_io_mem(app_ctx_.rknn_ctx, zero_copy_ctx_.output_mems[i], &zero_copy_ctx_.output_attrs[i]);
            if (ret < 0) {
                return false;
            }
        }
        
        input_mem_ = zero_copy_ctx_.input_mem;
        return true;
    }
    
    bool init_letterbox_context() {
        return init_letterbox_context(&zero_copy_ctx_.letterbox_ctx, 
                                    zero_copy_ctx_.model_width, 
                                    zero_copy_ctx_.model_height) == 0;
    }
    
    bool init_byte_tracker() {
        try {
            byte_tracker_ = std::make_unique<BYTETracker>(30, 30);  // fps=30, track_buffer=30
            return true;
        } catch (const std::exception& e) {
            printf("ByteTracker初始化失败: %s\n", e.what());
            return false;
        }
    }
    
    bool letterbox_resize_to_npu_memory(const cv::Mat& frame) {
        return letterbox_resize_to_npu(&zero_copy_ctx_.letterbox_ctx, 
                                     frame, 
                                     (uint8_t*)zero_copy_ctx_.input_mem->virt_addr,
                                     zero_copy_ctx_.input_attr.n_dims,
                                     zero_copy_ctx_.input_attr.dims) == 0;
    }
    
    void convert_detection_results(const object_detect_result_list& detect_results, 
                                 std::vector<PoseResult>& pose_results,
                                 const cv::Size& frame_size) {
        pose_results.clear();
        pose_results.reserve(detect_results.count);
        
        for (int i = 0; i < detect_results.count; i++) {
            const auto& obj = detect_results.results[i];
            if (obj.prop < conf_threshold_) continue;
            
            PoseResult result;
            result.person_id = -1;  // 将由跟踪器分配
            result.confidence = obj.prop;
            result.has_ground_position = false;
            
            // 转换边界框坐标 (letterbox -> 原图)
            result.bbox = convert_bbox_from_letterbox(obj.box, frame_size);
            
            // 转换关键点坐标
            result.keypoints.reserve(17);
            result.keypoint_scores.reserve(17);
            for (int j = 0; j < 17; j++) {
                cv::Point2f kpt = convert_keypoint_from_letterbox(obj.kps[j], frame_size);
                result.keypoints.push_back(kpt);
                result.keypoint_scores.push_back(obj.kps[j].score);
            }
            
            pose_results.push_back(result);
        }
    }
    
    cv::Rect convert_bbox_from_letterbox(const object_detect_result_box& box, const cv::Size& frame_size) {
        float scale_x = (float)frame_size.width / zero_copy_ctx_.model_width;
        float scale_y = (float)frame_size.height / zero_copy_ctx_.model_height;
        
        int x = (int)(box.left * scale_x);
        int y = (int)(box.top * scale_y);
        int w = (int)(box.right * scale_x) - x;
        int h = (int)(box.bottom * scale_y) - y;
        
        return cv::Rect(x, y, w, h);
    }
    
    cv::Point2f convert_keypoint_from_letterbox(const object_detect_result_keypoint& kpt, const cv::Size& frame_size) {
        float scale_x = (float)frame_size.width / zero_copy_ctx_.model_width;
        float scale_y = (float)frame_size.height / zero_copy_ctx_.model_height;
        
        return cv::Point2f(kpt.x * scale_x, kpt.y * scale_y);
    }
    
    void apply_byte_tracking(std::vector<PoseResult>& results) {
        if (!byte_tracker_ || results.empty()) return;
        
        // 转换为ByteTrack格式
        std::vector<Object> objects;
        for (size_t i = 0; i < results.size(); i++) {
            Object obj;
            obj.rect = cv::Rect2f(results[i].bbox);
            obj.prob = results[i].confidence;
            obj.label = 0;  // person类别
            objects.push_back(obj);
        }
        
        // 执行跟踪
        std::vector<STrack> output_stracks = byte_tracker_->update(objects);
        
        // 将跟踪ID分配给结果
        for (size_t i = 0; i < output_stracks.size() && i < results.size(); i++) {
            results[i].person_id = output_stracks[i].track_id;
        }
    }
    
    void apply_homography_mapping(std::vector<PoseResult>& results) {
        if (!has_homography_ || homography_matrix_.empty()) return;
        
        for (auto& result : results) {
            // 使用脚踝中点作为地面接触点
            if (result.keypoints.size() >= 17) {
                cv::Point2f left_ankle = result.keypoints[15];   // 左脚踝
                cv::Point2f right_ankle = result.keypoints[16];  // 右脚踝
                
                cv::Point2f ground_point = (left_ankle + right_ankle) * 0.5f;
                
                // 应用Homography变换
                std::vector<cv::Point2f> src_pts = {ground_point};
                std::vector<cv::Point2f> dst_pts;
                cv::perspectiveTransform(src_pts, dst_pts, homography_matrix_);
                
                result.ground_position = dst_pts[0];
                result.has_ground_position = true;
            }
        }
    }
    
    void cleanup_rknn_model() {
        if (app_ctx_.rknn_ctx) {
            release_yolov8_pose_model(&app_ctx_);
            memset(&app_ctx_, 0, sizeof(app_ctx_));
        }
    }
    
    void cleanup_zero_copy_memory() {
        for (int i = 0; i < 4; i++) {
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
    
    void cleanup_letterbox() {
        // letterbox上下文通常不需要特殊清理
    }
    
    void cleanup_byte_tracker() {
        byte_tracker_.reset();
    }
};

// PoseDetector公共接口实现
PoseDetector::PoseDetector(const std::string& model_path)
    : pImpl_(std::make_unique<Impl>(model_path)) {
}

PoseDetector::~PoseDetector() = default;

std::vector<PoseResult> PoseDetector::detect(const cv::Mat& frame) {
    return pImpl_->detect(frame);
}

void PoseDetector::enable_tracking(bool enable) {
    pImpl_->enable_tracking(enable);
}

bool PoseDetector::load_calibration(const std::string& calibration_file) {
    return pImpl_->load_calibration(calibration_file);
}

void PoseDetector::set_confidence_threshold(float threshold) {
    pImpl_->set_confidence_threshold(threshold);
}

bool PoseDetector::is_initialized() const {
    return pImpl_->is_initialized();
}

void PoseDetector::destroy() {
    pImpl_->cleanup_all_resources();
}