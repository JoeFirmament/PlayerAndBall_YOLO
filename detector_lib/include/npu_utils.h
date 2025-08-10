#ifndef NPU_UTILS_H
#define NPU_UTILS_H

#include <string>
#include <vector>
#include <fstream>
#include <sstream>

namespace detector {

/**
 * @brief NPU信息结构体
 */
struct NPUInfo {
    int total_cores = 0;           // NPU核心总数
    int available_cores = 0;       // 可用核心数
    std::vector<int> core_loads;   // 各核心负载率 (0-100)
    int current_freq_mhz = 0;      // 当前频率 (MHz)
    int temperature_celsius = 0;   // 温度 (摄氏度)
    std::string governor;          // 调频策略
};

/**
 * @brief NPU工具类
 */
class NPUUtils {
public:
    /**
     * @brief 获取NPU信息
     * @return NPU信息结构体
     */
    static NPUInfo get_npu_info() {
        NPUInfo info;
        
        // RK3588S有3个NPU核心
        info.total_cores = 3;
        
        // 获取频率
        info.current_freq_mhz = read_freq();
        
        // 获取温度
        info.temperature_celsius = read_temperature();
        
        // 获取调频策略
        info.governor = read_governor();
        
        // 获取负载（需要root权限）
        info.core_loads = read_loads();
        info.available_cores = info.total_cores; // 简化处理
        
        return info;
    }
    
    /**
     * @brief 检查NPU核心是否可用
     * @param core_id 核心ID (0, 1, 2)
     * @return true=可用, false=不可用或无效ID
     */
    static bool is_core_available(int core_id) {
        if (core_id < 0 || core_id > 2) {
            return false;
        }
        // 简化实现，实际应该检查设备节点
        return true;
    }
    
    /**
     * @brief 获取推荐的NPU核心
     * @param prefer_core 偏好核心 (-1=自动选择)
     * @return 推荐的核心ID
     */
    static int get_recommended_core(int prefer_core = -1) {
        if (prefer_core >= 0 && prefer_core <= 2 && is_core_available(prefer_core)) {
            return prefer_core;
        }
        
        // 自动选择策略：选择负载最低的核心
        auto info = get_npu_info();
        if (info.core_loads.empty()) {
            // 无负载信息，轮流分配
            static int next_core = 0;
            int selected = next_core;
            next_core = (next_core + 1) % 3;
            return selected;
        }
        
        // 选择负载最低的核心
        int min_load = 100;
        int best_core = 0;
        for (size_t i = 0; i < info.core_loads.size() && i < 3; i++) {
            if (info.core_loads[i] < min_load) {
                min_load = info.core_loads[i];
                best_core = i;
            }
        }
        
        return best_core;
    }
    
private:
    static int read_freq() {
        std::ifstream file("/sys/class/devfreq/fdab0000.npu/cur_freq");
        if (file.is_open()) {
            long freq;
            file >> freq;
            return freq / 1000000; // Hz to MHz
        }
        return 0;
    }
    
    static int read_temperature() {
        // 遍历thermal zones查找NPU
        for (int i = 0; i < 10; i++) {
            std::string type_path = "/sys/class/thermal/thermal_zone" + std::to_string(i) + "/type";
            std::ifstream type_file(type_path);
            if (type_file.is_open()) {
                std::string type;
                type_file >> type;
                if (type.find("npu") != std::string::npos) {
                    std::string temp_path = "/sys/class/thermal/thermal_zone" + std::to_string(i) + "/temp";
                    std::ifstream temp_file(temp_path);
                    if (temp_file.is_open()) {
                        int temp;
                        temp_file >> temp;
                        return temp / 1000; // millidegree to degree
                    }
                }
            }
        }
        return 0;
    }
    
    static std::string read_governor() {
        std::ifstream file("/sys/class/devfreq/fdab0000.npu/governor");
        if (file.is_open()) {
            std::string governor;
            file >> governor;
            return governor;
        }
        return "unknown";
    }
    
    static std::vector<int> read_loads() {
        std::vector<int> loads;
        
        // 需要root权限访问 /sys/kernel/debug/rknpu/load
        std::ifstream file("/sys/kernel/debug/rknpu/load");
        if (file.is_open()) {
            std::string line;
            while (std::getline(file, line)) {
                // 解析格式: NPU0: xx%, NPU1: xx%, NPU2: xx%
                size_t pos = line.find("NPU");
                while (pos != std::string::npos) {
                    size_t colon_pos = line.find(':', pos);
                    size_t percent_pos = line.find('%', colon_pos);
                    if (colon_pos != std::string::npos && percent_pos != std::string::npos) {
                        std::string load_str = line.substr(colon_pos + 1, percent_pos - colon_pos - 1);
                        int load = std::stoi(load_str);
                        loads.push_back(load);
                    }
                    pos = line.find("NPU", percent_pos);
                }
            }
        }
        
        // 如果无法读取，返回默认值
        if (loads.empty()) {
            loads = {0, 0, 0}; // 假设3个核心都空闲
        }
        
        return loads;
    }
};

} // namespace detector

#endif // NPU_UTILS_H