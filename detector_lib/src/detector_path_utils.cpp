#include "detector_path_utils.h"
#include <cstdlib>
#include <sys/stat.h>
#include <iostream>

namespace detector {

bool PathUtils::file_exists(const std::string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

std::string PathUtils::get_env_var(const std::string& var_name, const std::string& default_value) {
    const char* env_value = std::getenv(var_name.c_str());
    return env_value ? std::string(env_value) : default_value;
}

std::vector<std::string> PathUtils::get_model_search_paths() {
    std::vector<std::string> paths;
    
    // 1. 环境变量指定的路径（最高优先级）
    std::string env_path = get_env_var("DETECTOR_MODEL_PATH");
    if (!env_path.empty()) {
        paths.push_back(env_path);
    }
    
    // 2. 当前目录及相对路径
    paths.push_back("./models");
    paths.push_back("../models");
    paths.push_back("../../models");
    
    // 3. 系统安装路径
    paths.push_back("/usr/local/share/detector_lib/models");
    paths.push_back("/opt/detector_lib/models");
    
    // 4. 用户HOME目录
    std::string home = get_env_var("HOME");
    if (!home.empty()) {
        paths.push_back(home + "/.detector_lib/models");
        paths.push_back(home + "/detector_lib/models");
    }
    
    return paths;
}

std::vector<std::string> PathUtils::get_data_search_paths() {
    std::vector<std::string> paths;
    
    // 1. 环境变量指定的路径（最高优先级）
    std::string env_path = get_env_var("DETECTOR_DATA_PATH");
    if (!env_path.empty()) {
        paths.push_back(env_path);
    }
    
    // 2. 当前目录及相对路径
    paths.push_back("./data");
    paths.push_back("../data");
    paths.push_back("../../data");
    
    // 3. 系统安装路径
    paths.push_back("/usr/local/share/detector_lib/data");
    paths.push_back("/opt/detector_lib/data");
    
    // 4. 用户HOME目录
    std::string home = get_env_var("HOME");
    if (!home.empty()) {
        paths.push_back(home + "/.detector_lib/data");
        paths.push_back(home + "/detector_lib/data");
    }
    
    return paths;
}

std::string PathUtils::find_model(const std::string& model_name) {
    auto search_paths = get_model_search_paths();
    
    for (const auto& base_path : search_paths) {
        std::string full_path = base_path + "/" + model_name;
        if (file_exists(full_path)) {
            return full_path;
        }
    }
    
    
    
    return "";
}

std::string PathUtils::find_calibration(const std::string& calibration_name) {
    auto search_paths = get_data_search_paths();
    
    for (const auto& base_path : search_paths) {
        std::string full_path = base_path + "/" + calibration_name;
        if (file_exists(full_path)) {
            return full_path;
        }
    }
    
    
    
    return "";
}

} // namespace detector