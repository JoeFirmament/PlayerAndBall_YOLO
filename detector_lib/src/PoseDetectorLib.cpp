#include "PoseDetectorLib.h"
#include "internal/detector_common.h"
#include <cmath>
#include <set>
#include <algorithm>
#include <map>
#include <deque>
#include <fstream>
#include <opencv2/opencv.hpp>

// 添加ByteTrack支持（编译开关控制）
#ifdef DETECTOR_LIB_ENABLE_TRACKING
#include "BYTETracker.h"
#endif

// 添加Homography相关
#include <opencv2/calib3d.hpp>

// 添加姿态后处理需要的定义 (从pose_postprocess.h移植)
#define OBJ_CLASS_NUM 1
#define OBJ_NUMB_MAX_SIZE 128

typedef struct {
    image_rect_t box;
    float keypoints[17][3]; // keypoints x,y,conf
    float prop;
    int cls_id;
} object_detect_result;

typedef struct {
    int id;
    int count;
    object_detect_result results[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;

typedef struct {
    float x_pad;
    float y_pad;
    float scale;
} letterbox_t;

namespace detector {

// PoseDetectorLib内部实现类
class PoseDetectorLib::Impl {
public:
    explicit Impl(const std::string& model_path)
        : model_path_(model_path)
        , status_(DETECTOR_UNINITIALIZED)
        , tracking_enabled_(false)
        , conf_threshold_(0.1f)
        , has_homography_(false)
        , last_inference_time_ms_(-1)
        , homography_loaded_(false)
        , polar_enabled_(false)
        , polar_origin_offset_(0.0f, 0.0f)
    {
    }
    
    ~Impl() {
        cleanup();
    }
    
    std::vector<PoseResult> detect(const cv::Mat& frame) {
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
        
        // 3. 获取输出 - 完全按照工作版本方式
        rknn_output outputs[rknn_ctx_.io_num.n_output];
        memset(outputs, 0, sizeof(outputs));
        for (int i = 0; i < rknn_ctx_.io_num.n_output; i++) {
            outputs[i].index = i;
            outputs[i].want_float = (!rknn_ctx_.is_quant);
        }
        ret = rknn_outputs_get(rknn_ctx_.ctx, rknn_ctx_.io_num.n_output, outputs, NULL);
        if (ret < 0) {
            printf("获取输出失败! ret=%d\n", ret);
            return {};
        }
        
        // 4. 后处理 - 使用真正的pose_post_process函数
        std::vector<PoseResult> results = postprocess_real_results(outputs, frame.size(), scale, x_pad, y_pad);

        // 5. 可选：进行ByteTrack多目标跟踪，分配稳定ID（不改变ROI/关键点）
#ifdef DETECTOR_LIB_ENABLE_TRACKING
        if (tracking_enabled_ && !results.empty()) {
            std::vector<Object> objects;
            objects.reserve(results.size());
            for (const auto& r : results) {
                Object o;
                o.box = cv::Rect2f((float)r.bbox.x, (float)r.bbox.y,
                                   (float)r.bbox.width, (float)r.bbox.height);
                o.score = r.confidence;
                o.classId = 0; // person 类别
                objects.push_back(o);
            }

            std::vector<STrack> tracks = byte_tracker_.update(objects);

            // 初始化person_id为-1
            for (auto& r : results) r.person_id = -1;

            // 使用IoU将track回填到检测结果
            for (const auto& t : tracks) {
                cv::Rect2f tr(t.tlbr[0], t.tlbr[1], t.tlbr[2] - t.tlbr[0], t.tlbr[3] - t.tlbr[1]);
                int best_idx = -1;
                float best_iou = 0.0f;
                for (int i = 0; i < (int)results.size(); ++i) {
                    const auto& b = results[i].bbox;
                    cv::Rect2f det((float)b.x, (float)b.y, (float)b.width, (float)b.height);
                    // 计算IoU（简易版）
                    float x1 = std::max(det.x, tr.x);
                    float y1 = std::max(det.y, tr.y);
                    float x2 = std::min(det.x + det.width, tr.x + tr.width);
                    float y2 = std::min(det.y + det.height, tr.y + tr.height);
                    float inter = (x2 > x1 && y2 > y1) ? (x2 - x1) * (y2 - y1) : 0.0f;
                    float uni = det.width * det.height + tr.width * tr.height - inter;
                    float iou = uni > 0.0f ? inter / uni : 0.0f;
                    if (iou > best_iou) { best_iou = iou; best_idx = i; }
                }
                if (best_idx >= 0 && best_iou > 0.1f) {
                    results[best_idx].person_id = t.track_id;
                }
            }
        }
#endif
        
        // 释放输出
        rknn_outputs_release(rknn_ctx_.ctx, rknn_ctx_.io_num.n_output, outputs);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        last_inference_time_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        
        return results;
    }
    
    void enable_tracking(bool enable) {
        tracking_enabled_ = enable;
    }
    
    bool load_calibration(const std::string& calibration_file) {
        if (!internal::file_exists(calibration_file)) {
            return false;
        }
        
        // 读取JSON格式的Homography标定文件
        try {
            cv::FileStorage fs(calibration_file, cv::FileStorage::READ);
            if (!fs.isOpened()) {
                return false;
            }
            
            // 读取homography矩阵节点（支持多种字段名）
            cv::FileNode homo_node = fs["homography_matrix"];
            if (homo_node.empty()) {
                homo_node = fs["matrix"];  // 尝试另一个字段名
            }
            if (homo_node.empty()) {
                fs.release();
                return false;
            }
            
            // 创建3x3矩阵并填充数据
            homography_matrix_ = cv::Mat::zeros(3, 3, CV_64F);
            
            // 检查是否为数组格式（9个元素的一维数组）
            if (homo_node.type() == cv::FileNode::SEQ && homo_node.size() == 9) {
                // 处理一维数组格式 [h00, h01, h02, h10, h11, h12, h20, h21, h22]
                cv::FileNodeIterator it = homo_node.begin();
                for (int i = 0; i < 9 && it != homo_node.end(); i++, ++it) {
                    int row = i / 3;
                    int col = i % 3;
                    homography_matrix_.at<double>(row, col) = (double)*it;
                }
            } else {
                // 处理嵌套数组格式 [[h00, h01, h02], [h10, h11, h12], [h20, h21, h22]]
                int row = 0;
                for (cv::FileNodeIterator it = homo_node.begin(); it != homo_node.end(); ++it, ++row) {
                    cv::FileNode row_node = *it;
                    int col = 0;
                    for (cv::FileNodeIterator col_it = row_node.begin(); col_it != row_node.end(); ++col_it, ++col) {
                        homography_matrix_.at<double>(row, col) = (double)*col_it;
                    }
                }
            }
            
            fs.release();
            
            // 验证矩阵维度
            if (homography_matrix_.rows != 3 || homography_matrix_.cols != 3) {
                return false;
            }
            
            homography_loaded_ = true;
            has_homography_ = true;
            
            // 重新打开文件读取极坐标配置
            cv::FileStorage fs_polar(calibration_file, cv::FileStorage::READ);
            if (fs_polar.isOpened()) {
                // 读取原点偏移量（可选）
                cv::FileNode origin_node = fs_polar["origin_offset"];
                if (!origin_node.empty() && origin_node.isSeq() && origin_node.size() == 2) {
                    polar_origin_offset_.x = (float)origin_node[0];
                    polar_origin_offset_.y = (float)origin_node[1];
                } else {
                    polar_origin_offset_.x = 0.0f;
                    polar_origin_offset_.y = 0.0f;
                }
                
                // 读取极坐标开关（可选）
                cv::FileNode polar_node = fs_polar["use_polar"];
                if (!polar_node.empty()) {
                    polar_enabled_ = (int)polar_node != 0;
                } else {
                    polar_enabled_ = true;  // 默认开启
                }
                
                fs_polar.release();
            }
            
            return true;
        }
        catch (...) {
            return false;
        }
    }
    
    void set_confidence_threshold(float threshold) {
        conf_threshold_ = std::max(0.01f, std::min(0.99f, threshold));
    }
    
    void set_polar_coordinate_system(bool enable, float origin_offset_x, float origin_offset_y) {
        polar_enabled_ = enable;
        polar_origin_offset_.x = origin_offset_x;
        polar_origin_offset_.y = origin_offset_y;
    }
    
    // 笛卡尔坐标转极坐标
    PolarCoordinate cartesian_to_polar(const cv::Point2f& cartesian_point) const {
        PolarCoordinate polar;
        
        // 应用原点偏移
        double x = cartesian_point.x - polar_origin_offset_.x;
        double y = cartesian_point.y - polar_origin_offset_.y;
        
        // 计算极坐标
        polar.r = sqrt(x * x + y * y);           // 半径（距离）
        polar.theta = atan2(y, x);               // 角度（弧度，-π到π）
        
        return polar;
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
    bool tracking_enabled_;
    float conf_threshold_;
    bool has_homography_;
    int last_inference_time_ms_;
    
    // 内部资源
    internal::RknnContext rknn_ctx_;
    internal::ZeroCopyMemory zero_copy_mem_;
    
    // ByteTrack跟踪器（编译开关控制）
#ifdef DETECTOR_LIB_ENABLE_TRACKING
    BYTETracker byte_tracker_{}; // 使用默认参数: fps=30, buffer=30
#endif
    
    // Homography变换矩阵
    cv::Mat homography_matrix_;
    bool homography_loaded_;
    
    // 极坐标系统配置
    bool polar_enabled_;
    cv::Point2f polar_origin_offset_;
    
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
    
    bool preprocess_frame(const cv::Mat& frame) {
        if (!zero_copy_mem_.is_initialized) {
            return false;
        }
        
        // 简化的预处理：resize并拷贝到NPU内存
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(rknn_ctx_.model_width, rknn_ctx_.model_height));
        
        // 转换为RGB
        cv::Mat rgb_frame;
        cv::cvtColor(resized, rgb_frame, cv::COLOR_BGR2RGB);
        
        // 拷贝到NPU内存
        size_t data_size = rknn_ctx_.model_width * rknn_ctx_.model_height * rknn_ctx_.model_channel;
        memcpy(zero_copy_mem_.input_mem->virt_addr, rgb_frame.data, data_size);
        
        return true;
    }
    
    std::vector<PoseResult> postprocess_results(const cv::Size& frame_size) {
        // 简化的后处理实现
        // 在真实实现中，这里会解析NPU输出并生成PoseResult
        std::vector<PoseResult> results;
        
        // 模拟检测结果 (演示数据结构)
        PoseResult demo_result;
        demo_result.person_id = tracking_enabled_ ? 1 : -1;
        demo_result.confidence = 0.85f;
        demo_result.bbox = cv::Rect(100, 100, 200, 300);
        
        // 填充17个关键点 (模拟数据)
        demo_result.keypoints.resize(17);
        demo_result.keypoint_scores.resize(17);
        for (int i = 0; i < 17; i++) {
            demo_result.keypoints[i] = cv::Point2f(150 + i * 10, 150 + i * 15);
            demo_result.keypoint_scores[i] = 0.7f;
        }
        
        // 地面坐标映射 (如果有标定)
        if (has_homography_) {
            demo_result.ground_position = cv::Point2f(500.0f, 300.0f);
            demo_result.has_ground_position = true;
            
            // 极坐标计算
            if (polar_enabled_) {
                demo_result.polar_position = cartesian_to_polar(demo_result.ground_position);
                demo_result.has_polar_position = true;
            } else {
                demo_result.has_polar_position = false;
            }
        }
        
        // 只在置信度足够时添加结果
        if (demo_result.confidence >= conf_threshold_) {
            results.push_back(demo_result);
        }
        
        return results;
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
    
    // 真正的姿态检测后处理函数 - 严格按照pose_postprocess.cc实现
    std::vector<PoseResult> postprocess_real_results(rknn_output* outputs, const cv::Size& frame_size, float scale, int x_pad, int y_pad) {
        std::vector<PoseResult> results;
        
        
        
        // 构建letterbox参数
        letterbox_t letterbox;
        letterbox.x_pad = x_pad;
        letterbox.y_pad = y_pad;
        letterbox.scale = scale;
        
        // 调用真正的姿态后处理函数
        object_detect_result_list pose_results;
        int ret = real_pose_post_process(outputs, &letterbox, conf_threshold_, 0.4f, &pose_results);
        
        if (ret != 0) {
            printf("❌ 姿态后处理失败! ret=%d\n", ret);
            return results;
        }
        
        // 暂时跳过ByteTrack处理，直接使用检测结果
        
        // 基于检测结果生成输出（支持Homography坐标映射）
        for (int i = 0; i < pose_results.count; i++) {
            object_detect_result* result = &(pose_results.results[i]);
            
            PoseResult pose_result;
            pose_result.person_id = tracking_enabled_ ? (i + 1) : -1;  // 简化的ID分配
            pose_result.confidence = result->prop;
            
            // 边界框
            pose_result.bbox = cv::Rect(result->box.left, result->box.top,
                                       result->box.right - result->box.left,
                                       result->box.bottom - result->box.top);
            
            // 17个关键点
            pose_result.keypoints.resize(17);
            pose_result.keypoint_scores.resize(17);
            for (int j = 0; j < 17; j++) {
                pose_result.keypoints[j] = cv::Point2f(result->keypoints[j][0], result->keypoints[j][1]);
                pose_result.keypoint_scores[j] = result->keypoints[j][2];
            }
            
            // 计算地面坐标 - 使用ROI框底部中点
            if (homography_loaded_) {
                // 直接使用ROI框底部中点作为脚部位置
                cv::Point2f roi_bottom_center(
                    pose_result.bbox.x + pose_result.bbox.width / 2.0f,
                    pose_result.bbox.y + pose_result.bbox.height
                );
                
                pose_result.ground_position = apply_homography(roi_bottom_center);
                pose_result.has_ground_position = true;
                
                // 极坐标计算
                if (polar_enabled_) {
                    pose_result.polar_position = cartesian_to_polar(pose_result.ground_position);
                    pose_result.has_polar_position = true;
                } else {
                    pose_result.has_polar_position = false;
                }
            } else {
                pose_result.ground_position = cv::Point2f(-1, -1);
                pose_result.has_ground_position = false;
                pose_result.has_polar_position = false;
            }
            
            results.push_back(pose_result);
        }
        
        return results;
    }
    
    // 严格按照pose_postprocess.cc实现的真正后处理函数
    int real_pose_post_process(rknn_output* _outputs, letterbox_t* letter_box, float conf_threshold, float nms_threshold, object_detect_result_list* od_results) {
        
        std::vector<float> filterBoxes;
        std::vector<float> objProbs;
        std::vector<int> classId;
        int validCount = 0;
        int stride = 0;
        int grid_h = 0;
        int grid_w = 0;
        int model_in_w = rknn_ctx_.model_width;   // 640
        int model_in_h = rknn_ctx_.model_height;  // 640
        memset(od_results, 0, sizeof(object_detect_result_list));
        int index = 0;
        
        
        
        // 处理前3个输出张量（边界框检测） - 严格按照pose_postprocess.cc
        for (int i = 0; i < 3; i++) {
            grid_h = rknn_ctx_.output_attrs[i].dims[2];
            grid_w = rknn_ctx_.output_attrs[i].dims[3];
            stride = model_in_h / grid_h;
            
            
            
            if (rknn_ctx_.is_quant) {
                validCount += process_i8((int8_t*)_outputs[i].buf, grid_h, grid_w, stride, 
                                        filterBoxes, objProbs, classId, conf_threshold,
                                        rknn_ctx_.output_attrs[i].zp, rknn_ctx_.output_attrs[i].scale, index);
            } else {
                validCount += process_fp32((float*)_outputs[i].buf, grid_h, grid_w, stride, 
                                          filterBoxes, objProbs, classId, conf_threshold,
                                          rknn_ctx_.output_attrs[i].zp, rknn_ctx_.output_attrs[i].scale, index);
            }
            index += grid_h * grid_w;
        }
        
        
        
        // 如果没有检测到目标
        if (validCount <= 0) {
            return 0;
        }
        
        // NMS处理 - 严格按照pose_postprocess.cc
        std::vector<int> indexArray;
        for (int i = 0; i < validCount; ++i) {
            indexArray.push_back(i);
        }
        quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);
        
        std::set<int> class_set(std::begin(classId), std::end(classId));
        for (auto c : class_set) {
            nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
        }
        
        
        
        int last_count = 0;
        od_results->count = 0;
        
        // 提取最终检测结果和关键点 - 严格按照pose_postprocess.cc
        for (int i = 0; i < validCount; ++i) {
            if (indexArray[i] == -1 || last_count >= 128) {
                continue;
            }
            int n = indexArray[i];
            float x1 = filterBoxes[n * 5 + 0] - letter_box->x_pad;
            float y1 = filterBoxes[n * 5 + 1] - letter_box->y_pad;
            float w = filterBoxes[n * 5 + 2];
            float h = filterBoxes[n * 5 + 3];
            int keypoints_index = (int)filterBoxes[n * 5 + 4];
            
            // 提取17个关键点 - 严格按照pose_postprocess.cc
            for (int j = 0; j < 17; ++j) {
                if (rknn_ctx_.is_quant) {
                    // 量化版本的关键点提取
                    od_results->results[last_count].keypoints[j][0] = (deqnt_affine_to_f32(((int8_t*)_outputs[3].buf)[j*3*8400+0*8400+keypoints_index],
                            rknn_ctx_.output_attrs[3].zp, rknn_ctx_.output_attrs[3].scale) - letter_box->x_pad) / letter_box->scale;
                    od_results->results[last_count].keypoints[j][1] = (deqnt_affine_to_f32(((int8_t*)_outputs[3].buf)[j*3*8400+1*8400+keypoints_index],
                            rknn_ctx_.output_attrs[3].zp, rknn_ctx_.output_attrs[3].scale) - letter_box->y_pad) / letter_box->scale;
                    od_results->results[last_count].keypoints[j][2] = deqnt_affine_to_f32(((int8_t*)_outputs[3].buf)[j*3*8400+2*8400+keypoints_index],
                            rknn_ctx_.output_attrs[3].zp, rknn_ctx_.output_attrs[3].scale);
                } else {
                    // 非量化版本的关键点提取
                    od_results->results[last_count].keypoints[j][0] = (((float*)_outputs[3].buf)[j*3*8400+0*8400+keypoints_index] 
                                                                    - letter_box->x_pad) / letter_box->scale;
                    od_results->results[last_count].keypoints[j][1] = (((float*)_outputs[3].buf)[j*3*8400+1*8400+keypoints_index] 
                                                                        - letter_box->y_pad) / letter_box->scale;
                    od_results->results[last_count].keypoints[j][2] = ((float*)_outputs[3].buf)[j*3*8400+2*8400+keypoints_index];
                }
            }
            
            int id = classId[n];
            float obj_conf = objProbs[i];
            od_results->results[last_count].box.left = (int)(clamp(x1, 0, model_in_w) / letter_box->scale);
            od_results->results[last_count].box.top = (int)(clamp(y1, 0, model_in_h) / letter_box->scale);
            od_results->results[last_count].box.right = (int)(clamp(x1+w, 0, model_in_w) / letter_box->scale);
            od_results->results[last_count].box.bottom = (int)(clamp(y1+h, 0, model_in_h) / letter_box->scale);
            od_results->results[last_count].prop = obj_conf;
            od_results->results[last_count].cls_id = id;
            last_count++;
        }
        od_results->count = last_count;
        
        
        return 0;
    }
    
    // 真正的Float16转Float32函数 (从pose_postprocess.cc移植)
    float fp16_to_float(uint16_t fp16_val) {
        union { uint16_t i; float f; } converter;
        converter.i = fp16_val;
        return converter.f;
    }
    
    // 真正的边界框处理函数 (简化版本的process_fp32)
    int process_fp32_boxes(float *input, int grid_h, int grid_w, int stride,
                          std::vector<float> &boxes, std::vector<float> &boxScores, 
                          std::vector<int> &classId, float threshold, int32_t zp, float scale, int index) {
        int input_loc_len = 64;
        int validCount = 0;
        float thres_fp = unsigmoid(threshold);
        
        for (int h = 0; h < grid_h; h++) {
            for (int w = 0; w < grid_w; w++) {
                // 只检查第一个类别（person=0）
                if (input[(input_loc_len + 0) * grid_w * grid_h + h * grid_w + w] >= thres_fp) {
                    float box_conf_f32 = sigmoid(input[(input_loc_len + 0) * grid_w * grid_h + h * grid_w + w]);
                    
                    // DFL边界框回归
                    float loc[input_loc_len];
                    for (int i = 0; i < input_loc_len; ++i) {
                        loc[i] = input[i * grid_w * grid_h + h * grid_w + w];
                    }
                    
                    for (int i = 0; i < input_loc_len / 16; ++i) {
                        softmax(&loc[i * 16], 16);
                    }
                    
                    float xywh_[4] = {0, 0, 0, 0};
                    for (int dfl = 0; dfl < 16; ++dfl) {
                        xywh_[0] += loc[dfl] * dfl;
                        xywh_[1] += loc[1 * 16 + dfl] * dfl;
                        xywh_[2] += loc[2 * 16 + dfl] * dfl;
                        xywh_[3] += loc[3 * 16 + dfl] * dfl;
                    }
                    
                    // 转换为最终坐标
                    xywh_[0] = (w + 0.5) - xywh_[0];
                    xywh_[1] = (h + 0.5) - xywh_[1];
                    xywh_[2] = (w + 0.5) + xywh_[2];
                    xywh_[3] = (h + 0.5) + xywh_[3];
                    
                    float final_x = ((xywh_[0] + xywh_[2]) / 2) * stride;
                    float final_y = ((xywh_[1] + xywh_[3]) / 2) * stride;
                    float final_w = (xywh_[2] - xywh_[0]) * stride;
                    float final_h = (xywh_[3] - xywh_[1]) * stride;
                    
                    boxes.push_back(final_x - final_w/2);
                    boxes.push_back(final_y - final_h/2);
                    boxes.push_back(final_w);
                    boxes.push_back(final_h);
                    boxes.push_back(float(index + (h * grid_w) + w)); // keypoints index
                    boxScores.push_back(box_conf_f32);
                    classId.push_back(0); // person类别
                    validCount++;
                }
            }
        }
        return validCount;
    }
    
    // 量化反量化辅助函数
    float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) {
        return ((float)qnt - (float)zp) * scale;
    }
    
    int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale) {
        float dst_val = (f32 / scale) + zp;
        return (int8_t)std::max(-128.0f, std::min(127.0f, dst_val));
    }
    
    // int8边界框处理 (真正的实现，从pose_postprocess.cc移植)
    int process_int8_boxes(int8_t *input, int grid_h, int grid_w, int stride,
                          std::vector<float> &boxes, std::vector<float> &boxScores, 
                          std::vector<int> &classId, float threshold, int32_t zp, float scale, int index) {
        int input_loc_len = 64;
        int validCount = 0;
        
        int8_t thres_i8 = qnt_f32_to_affine(unsigmoid(threshold), zp, scale);
        
        
        for (int h = 0; h < grid_h; h++) {
            for (int w = 0; w < grid_w; w++) {
                // 检查第一个类别（person=0）的置信度
                int8_t conf_i8 = input[(input_loc_len + 0) * grid_w * grid_h + h * grid_w + w];
                
                if (conf_i8 >= thres_i8) {
                    float box_conf_f32 = sigmoid(deqnt_affine_to_f32(conf_i8, zp, scale));
                    
                    
                    
                    // DFL边界框回归 - 处理int8数据
                    float loc[input_loc_len];
                    for (int i = 0; i < input_loc_len; ++i) {
                        loc[i] = deqnt_affine_to_f32(input[i * grid_w * grid_h + h * grid_w + w], zp, scale);
                    }
                    
                    for (int i = 0; i < input_loc_len / 16; ++i) {
                        softmax(&loc[i * 16], 16);
                    }
                    
                    float xywh_[4] = {0, 0, 0, 0};
                    for (int dfl = 0; dfl < 16; ++dfl) {
                        xywh_[0] += loc[dfl] * dfl;
                        xywh_[1] += loc[1 * 16 + dfl] * dfl;
                        xywh_[2] += loc[2 * 16 + dfl] * dfl;
                        xywh_[3] += loc[3 * 16 + dfl] * dfl;
                    }
                    
                    // 转换为最终坐标
                    xywh_[0] = (w + 0.5) - xywh_[0];
                    xywh_[1] = (h + 0.5) - xywh_[1];
                    xywh_[2] = (w + 0.5) + xywh_[2];
                    xywh_[3] = (h + 0.5) + xywh_[3];
                    
                    float final_x = ((xywh_[0] + xywh_[2]) / 2) * stride;
                    float final_y = ((xywh_[1] + xywh_[3]) / 2) * stride;
                    float final_w = (xywh_[2] - xywh_[0]) * stride;
                    float final_h = (xywh_[3] - xywh_[1]) * stride;
                    
                    boxes.push_back(final_x - final_w/2);
                    boxes.push_back(final_y - final_h/2);
                    boxes.push_back(final_w);
                    boxes.push_back(final_h);
                    boxes.push_back(float(index + (h * grid_w) + w)); // keypoints index
                    boxScores.push_back(box_conf_f32);
                    classId.push_back(0); // person类别
                    validCount++;
                }
            }
        }
        
        return validCount;
    }
    
    // Float16边界框处理 (真正的实现)
    int process_fp16_boxes(void *input, int grid_h, int grid_w, int stride,
                          std::vector<float> &boxes, std::vector<float> &boxScores, 
                          std::vector<int> &classId, float threshold, int32_t zp, float scale, int index) {
        uint16_t* fp16_data = (uint16_t*)input;
        int input_loc_len = 64;
        int validCount = 0;
        float thres_fp = unsigmoid(threshold);
        
        
        
        for (int h = 0; h < grid_h; h++) {
            for (int w = 0; w < grid_w; w++) {
                // 检查第一个类别（person=0）的置信度
                float conf_raw = fp16_to_float(fp16_data[(input_loc_len + 0) * grid_w * grid_h + h * grid_w + w]);
                
                if (conf_raw >= thres_fp) {
                    float box_conf_f32 = sigmoid(conf_raw);
                    
                    
                    
                    // DFL边界框回归 - 处理Float16数据
                    float loc[input_loc_len];
                    for (int i = 0; i < input_loc_len; ++i) {
                        loc[i] = fp16_to_float(fp16_data[i * grid_w * grid_h + h * grid_w + w]);
                    }
                    
                    for (int i = 0; i < input_loc_len / 16; ++i) {
                        softmax(&loc[i * 16], 16);
                    }
                    
                    float xywh_[4] = {0, 0, 0, 0};
                    for (int dfl = 0; dfl < 16; ++dfl) {
                        xywh_[0] += loc[dfl] * dfl;
                        xywh_[1] += loc[1 * 16 + dfl] * dfl;
                        xywh_[2] += loc[2 * 16 + dfl] * dfl;
                        xywh_[3] += loc[3 * 16 + dfl] * dfl;
                    }
                    
                    // 转换为最终坐标
                    xywh_[0] = (w + 0.5) - xywh_[0];
                    xywh_[1] = (h + 0.5) - xywh_[1];
                    xywh_[2] = (w + 0.5) + xywh_[2];
                    xywh_[3] = (h + 0.5) + xywh_[3];
                    
                    float final_x = ((xywh_[0] + xywh_[2]) / 2) * stride;
                    float final_y = ((xywh_[1] + xywh_[3]) / 2) * stride;
                    float final_w = (xywh_[2] - xywh_[0]) * stride;
                    float final_h = (xywh_[3] - xywh_[1]) * stride;
                    
                    boxes.push_back(final_x - final_w/2);
                    boxes.push_back(final_y - final_h/2);
                    boxes.push_back(final_w);
                    boxes.push_back(final_h);
                    boxes.push_back(float(index + (h * grid_w) + w)); // keypoints index
                    boxScores.push_back(box_conf_f32);
                    classId.push_back(0); // person类别
                    validCount++;
                }
            }
        }
        
        return validCount;
    }
    
    // 严格按照pose_postprocess.cc移植的辅助函数
    inline int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }
    
    float sigmoid(float x) { return 1.0 / (1.0 + expf(-x)); }
    float unsigmoid(float y) { return -1.0 * logf((1.0 / y) - 1.0); }
    
    void softmax(float *input, int size) {
        float max_val = input[0];
        for (int i = 1; i < size; ++i) {
            if (input[i] > max_val) {
                max_val = input[i];
            }
        }
        
        float sum_exp = 0.0;
        for (int i = 0; i < size; ++i) {
            sum_exp += expf(input[i] - max_val);
        }
        
        for (int i = 0; i < size; ++i) {
            input[i] = expf(input[i] - max_val) / sum_exp;
        }
    }
    
    // 快速排序 - 从pose_postprocess.cc移植
    int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices) {
        if (left >= right) return left;
        
        float key = input[left];
        int key_index = indices[left];
        int low = left;
        int high = right;
        
        while (low < high) {
            while (low < high && input[high] <= key) {
                high--;
            }
            input[low] = input[high];
            indices[low] = indices[high];
            while (low < high && input[low] >= key) {
                low++;
            }
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;
        quick_sort_indice_inverse(input, left, low - 1, indices);
        quick_sort_indice_inverse(input, low + 1, right, indices);
        return low;
    }
    
    // NMS算法 - 从pose_postprocess.cc移植
    float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1, float ymax1) {
        float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
        float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
        float i = w * h;
        float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
        return u <= 0.f ? 0.f : (i / u);
    }
    
    int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order,
                   int filterId, float threshold) {
        for (int i = 0; i < validCount; ++i) {
            int n = order[i];
            if (n == -1 || classIds[n] != filterId) {
                continue;
            }
            for (int j = i + 1; j < validCount; ++j) {
                int m = order[j];
                if (m == -1 || classIds[m] != filterId) {
                    continue;
                }
                float xmin0 = outputLocations[n * 5 + 0];
                float ymin0 = outputLocations[n * 5 + 1];
                float xmax0 = outputLocations[n * 5 + 0] + outputLocations[n * 5 + 2];
                float ymax0 = outputLocations[n * 5 + 1] + outputLocations[n * 5 + 3];

                float xmin1 = outputLocations[m * 5 + 0];
                float ymin1 = outputLocations[m * 5 + 1];
                float xmax1 = outputLocations[m * 5 + 0] + outputLocations[m * 5 + 2];
                float ymax1 = outputLocations[m * 5 + 1] + outputLocations[m * 5 + 3];

                float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

                if (iou > threshold) {
                    order[j] = -1;
                }
            }
        }
        return 0;
    }
    
    // int8量化处理函数 - 从pose_postprocess.cc移植
    int process_i8(int8_t *input, int grid_h, int grid_w, int stride,
                          std::vector<float> &boxes, std::vector<float> &boxScores, std::vector<int> &classId, 
                          float threshold, int32_t zp, float scale, int index) {
        int input_loc_len = 64;
        int validCount = 0;
        
        int8_t thres_i8 = qnt_f32_to_affine(unsigmoid(threshold), zp, scale);
        
        for (int h = 0; h < grid_h; h++) {
            for (int w = 0; w < grid_w; w++) {
                for (int a = 0; a < OBJ_CLASS_NUM; a++) {
                    if (input[(input_loc_len + a) * grid_w * grid_h + h * grid_w + w] >= thres_i8) {
                        float box_conf_f32 = sigmoid(deqnt_affine_to_f32(input[(input_loc_len + a) * grid_w * grid_h + h * grid_w + w],
                                                     zp, scale));
                        float loc[input_loc_len];
                        for (int i = 0; i < input_loc_len; ++i) {
                            loc[i] = deqnt_affine_to_f32(input[i * grid_w * grid_h + h * grid_w + w], zp, scale);
                        }

                        for (int i = 0; i < input_loc_len / 16; ++i) {
                            softmax(&loc[i * 16], 16);
                        }
                        float xywh_[4] = {0, 0, 0, 0};
                        float xywh[4] = {0, 0, 0, 0};
                        for (int dfl = 0; dfl < 16; ++dfl) {
                            xywh_[0] += loc[dfl] * dfl;
                            xywh_[1] += loc[1 * 16 + dfl] * dfl;
                            xywh_[2] += loc[2 * 16 + dfl] * dfl;
                            xywh_[3] += loc[3 * 16 + dfl] * dfl;
                        }
                        xywh_[0] = (w + 0.5) - xywh_[0];
                        xywh_[1] = (h + 0.5) - xywh_[1];
                        xywh_[2] = (w + 0.5) + xywh_[2];
                        xywh_[3] = (h + 0.5) + xywh_[3];
                        xywh[0] = ((xywh_[0] + xywh_[2]) / 2) * stride;
                        xywh[1] = ((xywh_[1] + xywh_[3]) / 2) * stride;
                        xywh[2] = (xywh_[2] - xywh_[0]) * stride;
                        xywh[3] = (xywh_[3] - xywh_[1]) * stride;
                        xywh[0] = xywh[0] - xywh[2] / 2;
                        xywh[1] = xywh[1] - xywh[3] / 2;
                        boxes.push_back(xywh[0]);//x
                        boxes.push_back(xywh[1]);//y
                        boxes.push_back(xywh[2]);//w
                        boxes.push_back(xywh[3]);//h
                        boxes.push_back(float(index + (h * grid_w) + w));//keypoints index
                        boxScores.push_back(box_conf_f32);
                        classId.push_back(a);
                        validCount++;
                    }
                }
            }
        }
        return validCount;
    }
    
    // process_fp32 - 从pose_postprocess.cc移植
    int process_fp32(float *input, int grid_h, int grid_w, int stride,
                    std::vector<float> &boxes, std::vector<float> &boxScores, std::vector<int> &classId, 
                    float threshold, int32_t zp, float scale, int index) {
        int input_loc_len = 64;
        int validCount = 0;
        float thres_fp = unsigmoid(threshold);
        
        for (int h = 0; h < grid_h; h++) {
            for (int w = 0; w < grid_w; w++) {
                for (int a = 0; a < OBJ_CLASS_NUM; a++) {
                    if (input[(input_loc_len + a) * grid_w * grid_h + h * grid_w + w] >= thres_fp) {
                        float box_conf_f32 = sigmoid(input[(input_loc_len + a) * grid_w * grid_h + h * grid_w + w]);
                        float loc[input_loc_len];
                        for (int i = 0; i < input_loc_len; ++i) {
                            loc[i] = input[i * grid_w * grid_h + h * grid_w + w];
                        }

                        for (int i = 0; i < input_loc_len / 16; ++i) {
                            softmax(&loc[i * 16], 16);
                        }
                        float xywh_[4] = {0, 0, 0, 0};
                        float xywh[4] = {0, 0, 0, 0};
                        for (int dfl = 0; dfl < 16; ++dfl) {
                            xywh_[0] += loc[dfl] * dfl;
                            xywh_[1] += loc[1 * 16 + dfl] * dfl;
                            xywh_[2] += loc[2 * 16 + dfl] * dfl;
                            xywh_[3] += loc[3 * 16 + dfl] * dfl;
                        }
                        xywh_[0] = (w + 0.5) - xywh_[0];
                        xywh_[1] = (h + 0.5) - xywh_[1];
                        xywh_[2] = (w + 0.5) + xywh_[2];
                        xywh_[3] = (h + 0.5) + xywh_[3];
                        xywh[0] = ((xywh_[0] + xywh_[2]) / 2) * stride;
                        xywh[1] = ((xywh_[1] + xywh_[3]) / 2) * stride;
                        xywh[2] = (xywh_[2] - xywh_[0]) * stride;
                        xywh[3] = (xywh_[3] - xywh_[1]) * stride;
                        xywh[0] = xywh[0] - xywh[2] / 2;
                        xywh[1] = xywh[1] - xywh[3] / 2;
                        boxes.push_back(xywh[0]);//x
                        boxes.push_back(xywh[1]);//y
                        boxes.push_back(xywh[2]);//w
                        boxes.push_back(xywh[3]);//h
                        boxes.push_back(float(index + (h * grid_w) + w));//keypoints index
                        boxScores.push_back(box_conf_f32);
                        classId.push_back(a);
                        validCount++;
                    }
                }
            }
        }
        return validCount;
    }
    
    // 快速排序
    int quick_sort_boxes(std::vector<float> &input, int left, int right, std::vector<int> &indices) {
        if (left >= right) return 0;
        float key = input[left];
        int key_index = indices[left];
        int low = left, high = right;
        
        while (low < high) {
            while (low < high && input[high] <= key) high--;
            input[low] = input[high];
            indices[low] = indices[high];
            while (low < high && input[low] >= key) low++;
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;
        quick_sort_boxes(input, left, low - 1, indices);
        quick_sort_boxes(input, low + 1, right, indices);
        return low;
    }
    
    // NMS算法
    int nms_boxes(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, 
                 std::vector<int> &order, int filterId, float threshold) {
        for (int i = 0; i < validCount; ++i) {
            int n = order[i];
            if (n == -1 || classIds[n] != filterId) continue;
            
            for (int j = i + 1; j < validCount; ++j) {
                int m = order[j];
                if (m == -1 || classIds[m] != filterId) continue;
                
                float xmin0 = outputLocations[n * 5 + 0];
                float ymin0 = outputLocations[n * 5 + 1];
                float xmax0 = outputLocations[n * 5 + 0] + outputLocations[n * 5 + 2];
                float ymax0 = outputLocations[n * 5 + 1] + outputLocations[n * 5 + 3];
                
                float xmin1 = outputLocations[m * 5 + 0];
                float ymin1 = outputLocations[m * 5 + 1];
                float xmax1 = outputLocations[m * 5 + 0] + outputLocations[m * 5 + 2];
                float ymax1 = outputLocations[m * 5 + 1] + outputLocations[m * 5 + 3];
                
                float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
                float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
                float i_area = w * h;
                float u_area = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + 
                              (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i_area;
                float iou = u_area <= 0.f ? 0.f : (i_area / u_area);
                
                if (iou > threshold) {
                    order[j] = -1;
                }
            }
        }
        return 0;
    }
    
    // 计算两个矩形的IoU
    float calculate_iou(const cv::Rect_<float>& rect1, const cv::Rect_<float>& rect2) {
        float x1 = std::max(rect1.x, rect2.x);
        float y1 = std::max(rect1.y, rect2.y);
        float x2 = std::min(rect1.x + rect1.width, rect2.x + rect2.width);
        float y2 = std::min(rect1.y + rect1.height, rect2.y + rect2.height);
        
        if (x2 <= x1 || y2 <= y1) return 0.0f;
        
        float intersection = (x2 - x1) * (y2 - y1);
        float union_area = rect1.width * rect1.height + rect2.width * rect2.height - intersection;
        
        return union_area > 0 ? intersection / union_area : 0.0f;
    }
    
    // 从关键点计算脚部位置（用于Homography映射）
    cv::Point2f calculate_foot_position(const std::vector<cv::Point2f>& keypoints, const std::vector<float>& scores) {
        if (keypoints.size() < 17) return cv::Point2f(-1, -1);
        
        // 使用脚踝关键点（COCO格式：左脚踝=15，右脚踝=16）
        cv::Point2f left_ankle = keypoints[15];
        cv::Point2f right_ankle = keypoints[16];
        float left_score = scores[15];
        float right_score = scores[16];
        
        // 优先使用置信度高的脚踝，或使用两脚踝中点
        if (left_score > 0.3 && right_score > 0.3) {
            // 两个脚踝都可信，使用中点
            return cv::Point2f((left_ankle.x + right_ankle.x) / 2.0f, 
                              (left_ankle.y + right_ankle.y) / 2.0f);
        } else if (left_score > 0.3) {
            return left_ankle;
        } else if (right_score > 0.3) {
            return right_ankle;
        }
        
        // 脚踝不可信，尝试使用膝盖+偏移量
        cv::Point2f left_knee = keypoints[13];
        cv::Point2f right_knee = keypoints[14];
        float left_knee_score = scores[13];
        float right_knee_score = scores[14];
        
        if (left_knee_score > 0.3 && right_knee_score > 0.3) {
            cv::Point2f knee_center((left_knee.x + right_knee.x) / 2.0f, 
                                   (left_knee.y + right_knee.y) / 2.0f);
            // 向下偏移估计脚部位置
            return cv::Point2f(knee_center.x, knee_center.y + 100);
        }
        
        return cv::Point2f(-1, -1);  // 无法确定脚部位置
    }
    
    // 应用Homography变换将图像坐标映射到地面坐标
    cv::Point2f apply_homography(const cv::Point2f& image_point) {
        if (!homography_loaded_ || homography_matrix_.empty()) {
            return cv::Point2f(-1, -1);
        }
        
        std::vector<cv::Point2f> src_points = {image_point};
        std::vector<cv::Point2f> dst_points;
        
        try {
            cv::perspectiveTransform(src_points, dst_points, homography_matrix_);
            return dst_points[0];
        } catch (...) {
            return cv::Point2f(-1, -1);
        }
    }
};

// PoseDetectorLib公共接口实现
PoseDetectorLib::PoseDetectorLib(const std::string& model_path)
    : pImpl_(std::make_unique<Impl>(model_path)) {
}

PoseDetectorLib::~PoseDetectorLib() = default;

std::vector<PoseResult> PoseDetectorLib::detect(const cv::Mat& frame) {
    return pImpl_->detect(frame);
}

void PoseDetectorLib::enable_tracking(bool enable) {
    pImpl_->enable_tracking(enable);
}

bool PoseDetectorLib::load_calibration(const std::string& calibration_file) {
    return pImpl_->load_calibration(calibration_file);
}

void PoseDetectorLib::set_polar_coordinate_system(bool enable, float origin_offset_x, float origin_offset_y) {
    pImpl_->set_polar_coordinate_system(enable, origin_offset_x, origin_offset_y);
}

void PoseDetectorLib::set_confidence_threshold(float threshold) {
    pImpl_->set_confidence_threshold(threshold);
}

bool PoseDetectorLib::is_initialized() const {
    return pImpl_->is_initialized();
}

DetectorStatus PoseDetectorLib::get_status() const {
    return pImpl_->get_status();
}

void PoseDetectorLib::release() {
    pImpl_->cleanup();
}

int PoseDetectorLib::get_last_inference_time_ms() const {
    return pImpl_->get_last_inference_time_ms();
}

} // namespace detector