#!/bin/bash

echo "========================================="
echo "      RK3588 NPU 实时监控"
echo "========================================="
echo "按 Ctrl+C 退出"
echo ""

# 检查是否有权限访问调试文件系统
if [ ! -f "/sys/kernel/debug/rknpu/load" ]; then
    echo "❌ 无法访问 /sys/kernel/debug/rknpu/load"
    echo "请使用 sudo 运行此脚本："
    echo "sudo $0"
    exit 1
fi

# 清屏并监控
while true; do
    clear
    echo "========================================="
    echo "      RK3588 NPU 实时监控"
    echo "      $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================="
    
    # NPU负载
    echo -e "\n📊 NPU负载率："
    cat /sys/kernel/debug/rknpu/load
    
    # NPU频率
    echo -e "\n⚡ NPU频率："
    if [ -f "/sys/class/devfreq/fdab0000.npu/cur_freq" ]; then
        cur_freq=$(cat /sys/class/devfreq/fdab0000.npu/cur_freq)
        echo "当前: $(($cur_freq/1000000)) MHz"
    fi
    
    # NPU温度
    echo -e "\n🌡️ NPU温度："
    for thermal in /sys/class/thermal/thermal_zone*/type; do
        if grep -q "npu" "$thermal" 2>/dev/null; then
            zone=$(dirname "$thermal")
            temp=$(cat "$zone/temp" 2>/dev/null)
            if [ ! -z "$temp" ]; then
                echo "$(($temp/1000))°C"
            fi
        fi
    done
    
    # 内存使用（如果可用）
    if [ -f "/proc/rknpu/mem" ]; then
        echo -e "\n💾 NPU内存："
        cat /proc/rknpu/mem | head -5
    fi
    
    # 使用NPU的进程
    echo -e "\n🔍 NPU进程："
    lsof /dev/rknpu* 2>/dev/null | grep -v "COMMAND" | awk '{print $1, $2}' | sort -u || echo "无"
    
    echo -e "\n========================================="
    echo "更新间隔: 1秒 | 按 Ctrl+C 退出"
    
    sleep 1
done