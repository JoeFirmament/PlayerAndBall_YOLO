/**
 * @file rim_basketball_90fps_batch.cpp
 * @brief 90fps摄像头批量推理示例 - 充分利用NPU资源
 * 
 * 核心思想：
 * 1. 摄像头90fps采集 (1280x960 MJPEG)
 * 2. 4帧拼接成2x2网格送入NPU批量推理
 * 3. 提升NPU使用率从11%到40%+
 * 4. 详细日志记录验证全帧率推理
 * 
 * 编译：g++ rim_basketball_90fps_batch.cpp -ldetector_lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio -o rim_basketball_90fps_batch
 */

#include "RimBasketballDetectorLib.h"
#include "detector_path_utils.h"
#include <opencv2/opencv.hpp>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <fstream>
#include <iomanip>

using namespace detector;
using namespace std::chrono;

class HighFrameRateBatchDetector {
private:
    static const int BATCH_SIZE = 4;
    static const int SINGLE_WIDTH = 640;
    static const int SINGLE_HEIGHT = 640;
    static const int BATCH_WIDTH = SINGLE_WIDTH * 2;   // 1280
    static const int BATCH_HEIGHT = SINGLE_HEIGHT * 2; // 1280
    static const int MAX_QUEUE_SIZE = 30; // 最大缓存帧数
    
    RimBasketballDetectorLib detector;
    
    // 线程安全队列
    std::queue<cv::Mat> frame_queue;
    std::queue<cv::Mat> batch_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    
    // 统计信息
    std::atomic<int> total_captured_frames{0};
    std::atomic<int> total_processed_frames{0};
    std::atomic<int> total_inferences{0};
    std::atomic<bool> running{true};
    
    // 日志文件
    std::ofstream log_file;
    std::mutex log_mutex;
    
    high_resolution_clock::time_point start_time;
    
public:
    HighFrameRateBatchDetector(const std::string& model_path, int npu_core = 1) 
        : detector(model_path, npu_core) {
        
        start_time = high_resolution_clock::now();
        
        // 打开日志文件
        auto now = system_clock::now();
        auto time_t = system_clock::to_time_t(now);
        std::stringstream ss;
        ss << "rim_basketball_90fps_batch_" 
           << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") 
           << ".log";
        
        log_file.open(ss.str());
        write_log("INFO", "批量推理系统启动");
        write_log("INFO", "模型路径: " + model_path);
        write_log("INFO", "NPU核心: " + std::to_string(npu_core));
        write_log("INFO", "批处理大小: " + std::to_string(BATCH_SIZE));
    }
    
    ~HighFrameRateBatchDetector() {
        running = false;
        if (log_file.is_open()) {
            write_final_statistics();
            log_file.close();
        }
    }
    
    bool init_camera(int camera_id = 2) {
        cv::VideoCapture cap;
        
        // 尝试打开摄像头
        write_log("INFO", "尝试打开摄像头ID: " + std::to_string(camera_id));
        
        if (!cap.open(camera_id)) {
            write_log("ERROR", "无法打开摄像头ID: " + std::to_string(camera_id));
            return false;
        }
        
        // 设置MJPEG编码格式
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        
        // 设置分辨率 1280x960
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 960);
        
        // 设置90fps
        cap.set(cv::CAP_PROP_FPS, 90);
        
        // 设置缓存大小为1，减少延迟
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
        
        // 验证设置
        double actual_fps = cap.get(cv::CAP_PROP_FPS);
        double actual_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        double actual_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        
        write_log("INFO", "摄像头实际设置:");
        write_log("INFO", "  分辨率: " + std::to_string((int)actual_width) + "x" + std::to_string((int)actual_height));
        write_log("INFO", "  帧率: " + std::to_string(actual_fps) + " fps");
        
        // 检查是否达到目标配置
        if (actual_fps < 85.0 || actual_width != 1280 || actual_height != 960) {
            write_log("WARNING", "摄像头未达到目标配置 (1280x960@90fps)");
            write_log("WARNING", "可能影响批量推理效果");
        }
        
        cap.release();
        return true;
    }
    
    void start_processing(int camera_id = 2) {
        write_log("INFO", "开始批量推理处理");
        
        // 启动采集线程
        std::thread capture_thread(&HighFrameRateBatchDetector::capture_loop, this, camera_id);
        
        // 启动批处理线程
        std::thread batch_thread(&HighFrameRateBatchDetector::batch_processing_loop, this);
        
        // 启动统计线程
        std::thread stats_thread(&HighFrameRateBatchDetector::statistics_loop, this);
        
        // 主线程显示控制
        write_log("INFO", "按ESC键退出程序");
        
        while (running) {
            int key = cv::waitKey(30);
            if (key == 27) { // ESC键
                write_log("INFO", "收到退出信号");
                running = false;
                break;
            }
        }
        
        // 等待线程结束
        queue_cv.notify_all();
        capture_thread.join();
        batch_thread.join();
        stats_thread.join();
        
        write_log("INFO", "批量推理系统关闭");
    }
    
private:
    void write_log(const std::string& level, const std::string& message) {
        std::lock_guard<std::mutex> lock(log_mutex);
        auto now = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(now - start_time);
        
        auto time_t = system_clock::to_time_t(system_clock::now());
        
        // 控制台输出
        std::cout << "[" << std::put_time(std::localtime(&time_t), "%H:%M:%S") 
                  << "." << std::setfill('0') << std::setw(3) << (duration.count() % 1000)
                  << "] [" << level << "] " << message << std::endl;
        
        // 文件输出
        if (log_file.is_open()) {
            log_file << "[" << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S")
                     << "." << std::setfill('0') << std::setw(3) << (duration.count() % 1000)
                     << "] [" << level << "] " << message << std::endl;
            log_file.flush();
        }
    }
    
    void capture_loop(int camera_id) {
        cv::VideoCapture cap(camera_id);
        
        // 重新配置摄像头（线程中再次确认）
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 960);
        cap.set(cv::CAP_PROP_FPS, 90);
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
        
        if (!cap.isOpened()) {
            write_log("ERROR", "采集线程：无法打开摄像头");
            running = false;
            return;
        }
        
        write_log("INFO", "采集线程启动 - 目标90fps");
        
        cv::Mat frame;
        auto last_fps_time = high_resolution_clock::now();
        int fps_counter = 0;
        
        while (running) {
            auto frame_start = high_resolution_clock::now();
            
            if (!cap.read(frame)) {
                write_log("ERROR", "读取帧失败");
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            
            total_captured_frames++;
            fps_counter++;
            
            // 检查队列长度，避免内存溢出
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                if (frame_queue.size() >= MAX_QUEUE_SIZE) {
                    frame_queue.pop(); // 丢弃最老的帧
                    write_log("WARNING", "队列满，丢弃旧帧 (当前: " + std::to_string(frame_queue.size()) + ")");
                }
                frame_queue.push(frame.clone());
            }
            queue_cv.notify_one();
            
            // 每秒统计一次实际采集帧率
            auto now = high_resolution_clock::now();
            auto elapsed = duration_cast<milliseconds>(now - last_fps_time);
            if (elapsed.count() >= 1000) {
                double actual_fps = fps_counter * 1000.0 / elapsed.count();
                write_log("INFO", "实际采集帧率: " + std::to_string(actual_fps) + " fps, 队列长度: " + std::to_string(frame_queue.size()));
                
                last_fps_time = now;
                fps_counter = 0;
            }
            
            // 控制采集节拍 (90fps = ~11.1ms间隔)
            auto frame_end = high_resolution_clock::now();
            auto frame_duration = duration_cast<microseconds>(frame_end - frame_start);
            auto target_duration = microseconds(11111); // 90fps
            
            if (frame_duration < target_duration) {
                std::this_thread::sleep_for(target_duration - frame_duration);
            }
        }
        
        cap.release();
        write_log("INFO", "采集线程结束");
    }
    
    void batch_processing_loop() {
        write_log("INFO", "批处理线程启动");
        
        std::vector<cv::Mat> batch_frames;
        batch_frames.reserve(BATCH_SIZE);
        
        auto last_inference_time = high_resolution_clock::now();
        int inference_counter = 0;
        
        while (running || !frame_queue.empty()) {
            // 收集批处理帧
            batch_frames.clear();
            
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_cv.wait(lock, [this] { return !frame_queue.empty() || !running; });
                
                // 尝试收集BATCH_SIZE帧
                while (batch_frames.size() < BATCH_SIZE && !frame_queue.empty()) {
                    batch_frames.push_back(frame_queue.front());
                    frame_queue.pop();
                }
            }
            
            if (batch_frames.empty()) continue;
            
            // 执行批量推理
            auto inference_start = high_resolution_clock::now();
            
            auto results = detect_batch(batch_frames);
            
            auto inference_end = high_resolution_clock::now();
            auto inference_duration = duration_cast<milliseconds>(inference_end - inference_start);
            
            total_inferences++;
            total_processed_frames += batch_frames.size();
            inference_counter++;
            
            write_log("INFO", "批量推理完成 - 处理帧数: " + std::to_string(batch_frames.size()) + 
                             ", 耗时: " + std::to_string(inference_duration.count()) + "ms" +
                             ", 检测数: " + std::to_string(results.size()));
            
            // 统计推理帧率
            auto now = high_resolution_clock::now();
            auto elapsed = duration_cast<milliseconds>(now - last_inference_time);
            if (elapsed.count() >= 5000) { // 每5秒统计
                double inference_fps = (inference_counter * BATCH_SIZE) * 1000.0 / elapsed.count();
                write_log("INFO", "推理处理帧率: " + std::to_string(inference_fps) + " fps");
                
                last_inference_time = now;
                inference_counter = 0;
            }
        }
        
        write_log("INFO", "批处理线程结束");
    }
    
    std::vector<RimBasketballResult> detect_batch(const std::vector<cv::Mat>& frames) {
        if (frames.empty()) return {};
        
        // 创建批量拼接图像
        cv::Mat batch_image = create_batch_image(frames);
        
        // 单次NPU推理
        auto raw_results = detector.detect(batch_image);
        
        // 解析结果到各个子区域
        return parse_batch_results(raw_results, frames.size());
    }
    
    cv::Mat create_batch_image(const std::vector<cv::Mat>& frames) {
        cv::Mat batch_image(BATCH_HEIGHT, BATCH_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
        
        for (size_t i = 0; i < frames.size() && i < BATCH_SIZE; i++) {
            cv::Mat resized_frame;
            cv::resize(frames[i], resized_frame, cv::Size(SINGLE_WIDTH, SINGLE_HEIGHT));
            
            // 计算拼接位置 (2x2网格)
            int row = i / 2;
            int col = i % 2;
            int x = col * SINGLE_WIDTH;
            int y = row * SINGLE_HEIGHT;
            
            resized_frame.copyTo(batch_image(cv::Rect(x, y, SINGLE_WIDTH, SINGLE_HEIGHT)));
        }
        
        return batch_image;
    }
    
    std::vector<RimBasketballResult> parse_batch_results(
        const std::vector<RimBasketballResult>& raw_results, int actual_batch_size) {
        
        std::vector<RimBasketballResult> parsed_results;
        parsed_results.reserve(raw_results.size());
        
        for (const auto& result : raw_results) {
            // 判断检测框属于哪个子区域
            int sub_image_idx = get_sub_image_index(result.bbox);
            
            if (sub_image_idx >= actual_batch_size) continue; // 超出实际批大小
            
            // 坐标转换：从批量图坐标转换为单帧坐标
            RimBasketballResult adjusted_result = result;
            adjust_coordinates_to_original(adjusted_result, sub_image_idx);
            
            parsed_results.push_back(adjusted_result);
        }
        
        return parsed_results;
    }
    
    int get_sub_image_index(const cv::Rect& bbox) {
        int center_x = bbox.x + bbox.width / 2;
        int center_y = bbox.y + bbox.height / 2;
        
        int col = center_x >= SINGLE_WIDTH ? 1 : 0;
        int row = center_y >= SINGLE_HEIGHT ? 1 : 0;
        
        return row * 2 + col;  // 0,1,2,3
    }
    
    void adjust_coordinates_to_original(RimBasketballResult& result, int sub_image_idx) {
        // 计算子图在批量图中的偏移
        int col = sub_image_idx % 2;
        int row = sub_image_idx / 2;
        int offset_x = col * SINGLE_WIDTH;
        int offset_y = row * SINGLE_HEIGHT;
        
        // 调整边界框坐标
        result.bbox.x -= offset_x;
        result.bbox.y -= offset_y;
        
        // 调整中心点坐标
        result.center.x -= offset_x;
        result.center.y -= offset_y;
        
        // 坐标缩放：从640x640转换回1280x960
        float scale_x = 1280.0f / SINGLE_WIDTH;  // 2.0
        float scale_y = 960.0f / SINGLE_HEIGHT;  // 1.5
        
        result.bbox.x = (int)(result.bbox.x * scale_x);
        result.bbox.y = (int)(result.bbox.y * scale_y);
        result.bbox.width = (int)(result.bbox.width * scale_x);
        result.bbox.height = (int)(result.bbox.height * scale_y);
        
        result.center.x *= scale_x;
        result.center.y *= scale_y;
    }
    
    void statistics_loop() {
        write_log("INFO", "统计线程启动");
        
        while (running) {
            std::this_thread::sleep_for(std::chrono::seconds(10));
            
            double runtime_seconds = duration_cast<seconds>(high_resolution_clock::now() - start_time).count();
            double capture_fps = total_captured_frames.load() / runtime_seconds;
            double process_fps = total_processed_frames.load() / runtime_seconds;
            double inference_rate = total_inferences.load() / runtime_seconds;
            
            write_log("STATS", "运行时间: " + std::to_string((int)runtime_seconds) + "s");
            write_log("STATS", "采集帧率: " + std::to_string(capture_fps) + " fps");
            write_log("STATS", "处理帧率: " + std::to_string(process_fps) + " fps");
            write_log("STATS", "推理频率: " + std::to_string(inference_rate) + " batch/s");
            write_log("STATS", "NPU利用率提升: " + std::to_string(process_fps / 22.5) + "x (理论)");
        }
        
        write_log("INFO", "统计线程结束");
    }
    
    void write_final_statistics() {
        double runtime_seconds = duration_cast<seconds>(high_resolution_clock::now() - start_time).count();
        
        write_log("FINAL", "=== 最终统计 ===");
        write_log("FINAL", "总运行时间: " + std::to_string(runtime_seconds) + " 秒");
        write_log("FINAL", "总采集帧数: " + std::to_string(total_captured_frames.load()));
        write_log("FINAL", "总处理帧数: " + std::to_string(total_processed_frames.load()));
        write_log("FINAL", "总推理次数: " + std::to_string(total_inferences.load()));
        write_log("FINAL", "平均采集帧率: " + std::to_string(total_captured_frames.load() / runtime_seconds) + " fps");
        write_log("FINAL", "平均处理帧率: " + std::to_string(total_processed_frames.load() / runtime_seconds) + " fps");
        write_log("FINAL", "推理效率提升: " + std::to_string((total_processed_frames.load() / runtime_seconds) / 22.5) + "x");
        write_log("FINAL", "帧处理成功率: " + std::to_string(100.0 * total_processed_frames.load() / total_captured_frames.load()) + "%");
    }
};

int main(int argc, char* argv[]) {
    std::cout << "=== RIM Basketball 90fps 批量推理示例 ===" << std::endl;
    std::cout << "目标：充分利用NPU资源，提升推理效率" << std::endl;
    std::cout << "策略：90fps采集 + 4帧批量推理" << std::endl;
    
    // 解析命令行参数
    std::string model_path = "../models/Q_Rim_Basketball_724_JZ.rknn";
    int camera_id = 2;
    int npu_core = 1;  // 使用NPU核心1
    
    if (argc >= 2) model_path = argv[1];
    if (argc >= 3) camera_id = std::atoi(argv[2]);
    if (argc >= 4) npu_core = std::atoi(argv[3]);
    
    std::cout << "模型路径: " << model_path << std::endl;
    std::cout << "摄像头ID: " << camera_id << std::endl;
    std::cout << "NPU核心: " << npu_core << std::endl;
    
    try {
        // 智能路径查找
        std::string resolved_model_path = PathUtils::find_model("Q_Rim_Basketball_724_JZ.rknn");
        if (!resolved_model_path.empty()) {
            model_path = resolved_model_path;
            std::cout << "找到模型: " << model_path << std::endl;
        }
        
        // 创建批量检测器
        HighFrameRateBatchDetector batch_detector(model_path, npu_core);
        
        // 初始化摄像头
        if (!batch_detector.init_camera(camera_id)) {
            std::cerr << "摄像头初始化失败！" << std::endl;
            return -1;
        }
        
        // 开始处理
        batch_detector.start_processing(camera_id);
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return -1;
    }
    
    std::cout << "程序结束" << std::endl;
    return 0;
}