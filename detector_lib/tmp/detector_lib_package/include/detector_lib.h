#ifndef DETECTOR_LIB_H
#define DETECTOR_LIB_H

/**
 * @file detector_lib.h
 * @brief YOLOv8检测器库的统一头文件
 * 
 * 这是检测器库的主要包含文件，用户只需要包含这一个头文件即可
 * 使用所有检测功能。
 * 
 * 支持的功能：
 * - YOLOv8姿态检测 (17个COCO关键点)
 * - ByteTrack多目标跟踪
 * - Homography坐标映射
 * - 篮筐和篮球检测
 * - ROI分析和距离计算
 * 
 * 使用示例：
 * @code
 * #include "detector_lib.h"
 * 
 * int main() {
 *     detector::PoseDetectorLib pose_detector("models/Q_yolov8_pose.rknn");
 *     detector::RimBasketballDetectorLib rim_detector("models/Q_Rim_Basketball_724_JZ.rknn");
 *     
 *     cv::Mat frame;
 *     auto pose_results = pose_detector.detect(frame);
 *     auto rim_results = rim_detector.detect(frame);
 *     
 *     return 0;
 * }
 * @endcode
 */

// 核心类型定义
#include "detector_types.h"

// 检测器接口
#include "PoseDetectorLib.h"
#include "RimBasketballDetectorLib.h"

/**
 * @namespace detector
 * @brief 检测器库命名空间
 * 
 * 包含所有检测器相关的类和类型定义
 */
namespace detector {

/**
 * @brief 库版本信息
 */
struct LibraryInfo {
    static const char* VERSION;        ///< 版本号
    static const char* BUILD_DATE;     ///< 编译日期
    static const char* PLATFORM;      ///< 目标平台
    static const char* DESCRIPTION;   ///< 库描述
};

/**
 * @brief 获取库版本信息
 * @return LibraryInfo 结构体
 */
LibraryInfo get_library_info();

/**
 * @brief 检查RKNN运行时环境
 * @return true=环境正常，false=环境异常
 * @note 建议在使用检测器前调用此函数检查环境
 */
bool check_runtime_environment();

/**
 * @brief 设置全局日志级别
 * @param level 日志级别：0=静默，1=错误，2=警告，3=信息，4=调试
 */
void set_log_level(int level);

} // namespace detector

#endif // DETECTOR_LIB_H