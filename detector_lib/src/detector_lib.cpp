#include "detector_lib.h"
#include "internal/detector_common.h"

namespace detector {

// 库版本信息实现
const char* LibraryInfo::VERSION = "1.0.4";
const char* LibraryInfo::BUILD_DATE = __DATE__ " " __TIME__;
const char* LibraryInfo::PLATFORM = "RK3588";
const char* LibraryInfo::DESCRIPTION = "YOLOv8 Detection Library for RK3588 NPU";

LibraryInfo get_library_info() {
    return LibraryInfo{};
}

bool check_runtime_environment() {
    
    // 检查RKNN设备权限
    const char* device_paths[] = {
        "/dev/dri/renderD128",
        "/dev/dri/renderD129",
        nullptr
    };
    
    bool has_device = false;
    for (int i = 0; device_paths[i] != nullptr; i++) {
        if (internal::file_exists(device_paths[i])) {
            has_device = true;
            break;
        }
    }
    
    if (!has_device) {
        return false;
    }
    
    // 检查库文件
    // TODO: 可以添加更多运行时检查
    
    return true;
}

void set_log_level(int level) {
    // 日志系统已移除 - 此接口保留用于兼容性
    (void)level; // 避免未使用警告
}

} // namespace detector