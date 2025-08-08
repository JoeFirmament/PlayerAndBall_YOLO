#include "PoseDetectorLib.h"
#include "detector_path_utils.h"
#include <opencv2/opencv.hpp>
#include <iostream>

// 摄像头实时检测示例：启用 ByteTrack，打印 person_id，并绘制 ROI 框
int main(int argc, char** argv) {
    std::string model_path = argc > 1 ? argv[1] : std::string("../models/Q_yolov8_pose.rknn");
    int cam_index = 0;
    if (argc > 2) {
        try { cam_index = std::stoi(argv[2]); } catch (...) { cam_index = 0; }
    }
    // 可选：第三参数为标定文件路径；若未提供则智能查找默认标定文件
    std::string calib_path = argc > 3 ? argv[3] : detector::PathUtils::find_calibration("2025_8_6_1280_720.json");

    cv::VideoCapture cap;
    // 强制V4L2后端，若失败回退默认
    if (!cap.open(cam_index, cv::CAP_V4L2)) {
        if (!cap.open(cam_index)) {
            std::cerr << "Failed to open camera index " << cam_index << std::endl;
            return -1;
        }
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(cv::CAP_PROP_FPS, 30);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));

    detector::PoseDetectorLib detector(model_path);
    detector.enable_tracking(true); // 启用ByteTrack
    detector.set_confidence_threshold(0.25f);
    // 加载Homography标定（若找到），并自动启用极坐标（由库内部根据JSON决定）
    if (!calib_path.empty()) {
        bool ok = detector.load_calibration(calib_path);
        std::cout << (ok ? "✓ Homography/极坐标 已加载: " : "⚠ Homography加载失败: ") << calib_path << std::endl;
    } else {
        std::cout << "⚠ 未提供或未找到标定文件，跳过地面/极坐标显示" << std::endl;
    }

    cv::Mat frame;
    while (true) {
        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "Camera read failed" << std::endl;
            break;
        }

        auto results = detector.detect(frame);

        // 绘制结果
        for (const auto& r : results) {
            cv::rectangle(frame, r.bbox, cv::Scalar(0, 255, 0), 2);
            char info[64];
            snprintf(info, sizeof(info), "ID:%d conf:%.2f", r.person_id, r.confidence);
            cv::putText(frame, info, cv::Point(r.bbox.x, std::max(0, r.bbox.y - 5)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

            // 画底部中点（ROI基准）
            cv::Point2f bottom_center(r.bbox.x + r.bbox.width * 0.5f, r.bbox.y + r.bbox.height);
            cv::circle(frame, bottom_center, 4, cv::Scalar(255, 0, 255), -1);

            int text_y = r.bbox.y + r.bbox.height + 18;
            // 显示地面坐标
            if (r.has_ground_position) {
                char world_txt[96];
                snprintf(world_txt, sizeof(world_txt), "World(mm): %.0f, %.0f", r.ground_position.x, r.ground_position.y);
                cv::putText(frame, world_txt, cv::Point(r.bbox.x, std::min(text_y, frame.rows - 5)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
                text_y += 18;
            }
            // 显示极坐标
            if (r.has_polar_position) {
                char polar_txt[96];
                // theta_degrees() 是在结构体里实现的转换函数；若无则按弧度转度
                float theta_deg = r.polar_position.theta_degrees();
                snprintf(polar_txt, sizeof(polar_txt), "Polar: r=%.0fmm, theta=%.1fdeg", r.polar_position.r, theta_deg);
                cv::putText(frame, polar_txt, cv::Point(r.bbox.x, std::min(text_y, frame.rows - 5)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
            }
        }

        cv::imshow("pose_camera", frame);
        int key = cv::waitKey(1) & 0xFF;
        if (key == 27) break; // ESC
    }

    return 0;
}


