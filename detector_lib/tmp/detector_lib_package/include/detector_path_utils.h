#ifndef DETECTOR_PATH_UTILS_H
#define DETECTOR_PATH_UTILS_H

#include <string>
#include <vector>

namespace detector {

/**
 * @brief 路径解析工具类
 * 
 * 提供灵活的文件路径解析，支持：
 * - 环境变量
 * - 相对路径搜索
 * - 默认路径回退
 */
class PathUtils {
public:
    /**
     * @brief 查找模型文件
     * @param model_name 模型文件名（如 "Q_yolov8_pose.rknn"）
     * @return 完整的模型文件路径，空字符串表示未找到
     */
    static std::string find_model(const std::string& model_name);
    
    /**
     * @brief 查找标定文件
     * @param calibration_name 标定文件名（如 "calibration.json"）
     * @return 完整的标定文件路径，空字符串表示未找到
     */
    static std::string find_calibration(const std::string& calibration_name);
    
    /**
     * @brief 获取模型搜索路径列表
     * @return 按优先级排序的搜索路径列表
     */
    static std::vector<std::string> get_model_search_paths();
    
    /**
     * @brief 获取标定文件搜索路径列表
     * @return 按优先级排序的搜索路径列表
     */
    static std::vector<std::string> get_data_search_paths();

private:
    static bool file_exists(const std::string& path);
    static std::string get_env_var(const std::string& var_name, const std::string& default_value = "");
};

/**
 * @brief 支持的环境变量
 * 
 * DETECTOR_MODEL_PATH: 模型文件目录
 * DETECTOR_DATA_PATH:  标定数据目录
 * DETECTOR_ROOT:       库根目录
 */

} // namespace detector

#endif // DETECTOR_PATH_UTILS_H