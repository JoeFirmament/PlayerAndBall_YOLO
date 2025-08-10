#!/bin/bash

echo "========================================="
echo "        RK3588 NPU 信息检测脚本"
echo "========================================="

# 检测NPU设备
echo -e "\n📊 NPU设备信息："
if [ -d "/sys/class/devfreq/fdab0000.npu" ]; then
    echo "✅ 检测到RKNPU设备"
    
    # 获取当前频率
    if [ -f "/sys/class/devfreq/fdab0000.npu/cur_freq" ]; then
        cur_freq=$(cat /sys/class/devfreq/fdab0000.npu/cur_freq)
        echo "当前频率: $(($cur_freq/1000000)) MHz"
    fi
    
    # 获取可用频率
    if [ -f "/sys/class/devfreq/fdab0000.npu/available_frequencies" ]; then
        echo "可用频率: "
        cat /sys/class/devfreq/fdab0000.npu/available_frequencies | tr ' ' '\n' | while read freq; do
            if [ ! -z "$freq" ]; then
                echo "  - $(($freq/1000000)) MHz"
            fi
        done
    fi
    
    # 获取调频策略
    if [ -f "/sys/class/devfreq/fdab0000.npu/governor" ]; then
        governor=$(cat /sys/class/devfreq/fdab0000.npu/governor)
        echo "调频策略: $governor"
    fi
else
    echo "❌ 未检测到NPU设备"
fi

# 检测RKNN驱动
echo -e "\n🔧 RKNN驱动信息："
if [ -f "/proc/rknpu/version" ]; then
    echo "RKNN驱动版本:"
    cat /proc/rknpu/version
else
    echo "❌ 未找到RKNN驱动版本信息"
fi

# 检测NPU内存信息
echo -e "\n💾 NPU内存信息："
if [ -d "/proc/rknpu" ]; then
    if [ -f "/proc/rknpu/mem" ]; then
        echo "NPU内存使用情况:"
        cat /proc/rknpu/mem | head -20
    fi
fi

# 检测NPU核心数
echo -e "\n🎯 NPU核心信息："
# RK3588理论上有3个NPU核心
npu_cores=0
for i in 0 1 2; do
    # 检查设备节点
    if [ -e "/dev/rknpu_dev$i" ] || [ -e "/dev/npu_device$i" ]; then
        echo "检测到NPU核心$i"
        ((npu_cores++))
    fi
done

if [ $npu_cores -eq 0 ]; then
    # 尝试从dmesg获取信息
    dmesg | grep -i "npu" | grep -i "core" | tail -5
fi

# 检测NPU负载（如果支持）
echo -e "\n📈 NPU负载信息："
if [ -f "/sys/kernel/debug/rknpu/load" ]; then
    echo "NPU负载:"
    cat /sys/kernel/debug/rknpu/load
elif [ -f "/proc/rknpu/load" ]; then
    echo "NPU负载:"
    cat /proc/rknpu/load
else
    echo "⚠️  无法获取NPU负载信息（可能需要root权限或调试支持）"
fi

# 检测温度
echo -e "\n🌡️ NPU温度信息："
for thermal in /sys/class/thermal/thermal_zone*/type; do
    if grep -q "npu" "$thermal" 2>/dev/null; then
        zone=$(dirname "$thermal")
        temp=$(cat "$zone/temp" 2>/dev/null)
        if [ ! -z "$temp" ]; then
            echo "NPU温度: $(($temp/1000))°C"
        fi
    fi
done

# 进程使用情况
echo -e "\n🔍 使用NPU的进程："
lsof /dev/rknpu* 2>/dev/null | grep -v "COMMAND" || echo "当前没有进程使用NPU"

echo -e "\n========================================="