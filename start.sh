#!/bin/bash

# YOLOv8篮球检测系统 - 主启动菜单
# 快速启动各种检测模式

clear
echo "========================================="
echo "    YOLOv8篮球检测系统 - 主菜单"
echo "========================================="
echo "请选择运行模式:"
echo ""
echo "1. 双摄像头双线程检测 (推荐)"
echo "   - 姿态检测 + 篮筐篮球检测"
echo "   - 双NPU并行处理"
echo ""
echo "2. 纯姿态检测"
echo "   - 单独运行姿态检测和跟踪"
echo "   - 支持坐标映射"
echo ""
echo "3. 篮筐篮球检测"
echo "   - 单独运行篮筐和篮球检测"
echo "   - ROI分析功能"
echo ""
echo "4. 编译所有程序"
echo "   - 重新编译整个项目"
echo ""
echo "5. 性能测试"
echo "   - RGA硬件加速性能测试"
echo ""
echo "0. 退出"
echo "========================================="

read -p "请输入选项 (0-5): " choice

case $choice in
    1)
        echo "启动双摄像头双线程检测系统..."
        ./run_dual_camera.sh
        ;;
    2)
        echo "启动纯姿态检测系统..."
        ./run_pose_only.sh
        ;;
    3)
        echo "启动篮筐篮球检测系统..."
        ./run_rim_basketball.sh
        ;;
    4)
        echo "编译所有程序..."
        cd build || exit 1
        echo "清理构建目录..."
        make clean
        echo "重新编译..."
        make -j$(nproc)
        if [ $? -eq 0 ]; then
            echo "✅ 编译完成！"
        else
            echo "❌ 编译失败！"
        fi
        cd ..
        ;;
    5)
        echo "启动RGA性能测试..."
        ./test_rga_performance.sh
        ;;
    0)
        echo "退出程序"
        exit 0
        ;;
    *)
        echo "❌ 无效选项，请重新选择"
        sleep 2
        exec "$0"
        ;;
esac