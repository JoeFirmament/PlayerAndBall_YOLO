### Detector Lib 待办清单（交付与结构调整）

- [ ] 打包结构分层：在交付包中新增 `runtime/` 与 `source/` 两个目录
  - [ ] runtime：`lib/、include/、models/、examples/、data/、imgs/、scripts/、docs/、install.sh、uninstall.sh`
  - [ ] source：`CMakeLists.txt、src/、include/、examples/、docs/、build_and_install.sh、data/、imgs`
  - [ ] 更新 `README_QUICK.md`（运行说明指向 `runtime/`，编译说明指向 `source/`）
  - [ ] 更新 `install.sh`/`uninstall.sh` 文案，文档引用改为 `docs/USER_GUIDE.md`

- [ ] 打包脚本改造：`create_distribution_package.sh`
  - [ ] 复制 runtime 内容到 `dist/detector_lib_package/runtime/`
  - [ ] 复制源码到 `dist/detector_lib_package/source/`
  - [ ] 包含 `librknnrt.so` 到 `runtime/lib/`（若存在）
  - [ ] 生成 `README_QUICK.md`（分层说明、示例命令）
  - [ ] 输出摘要中展示分层后的目录结构

- [ ] 示例完整性校验
  - [ ] `examples/pose_camera_bytetrack_homography` 必须存在于 `runtime/examples/`
  - [ ] 单图示例默认关闭 ByteTrack（已实现，文档需强调）
  - [ ] 可选：新增运行脚本 `scripts/run_pose_camera_bytetrack_homography.sh`

- [ ] 文档一致性
  - [ ] `README.md/USER_GUIDE.md/DetectorAPI_Usage.md/PACKAGE_INFO.md/CHANGELOG.md` 路径统一与分层说明
  - [ ] 故障排除：区分 RKNN 与 NCNN（避免 `find_blob_index_by_name` 误解）
  - [ ] 零拷贝属于内部实现，用户文档不强调；API 使用方式为主

- [ ] 平台与版本信息
  - [ ] 在交付包根目录保留 `VERSION`，构建时间/平台信息
  - [ ] 恢复/生成 `PLATFORM_LOG.md`（RK3588、Debian/内核、OpenCV/RKNN 版本等）
  - [ ] 下一版本号：`1.0.4`（完成分层后打 Tag）

- [ ] 清理与仓库健康
  - [ ] 确认内层旧结构已移除引用（不立即删除仓库文件，先检索引用）
  - [ ] `.gitignore` 忽略 `build/、dist/、*_result.jpg、*.tar.gz、*.md5`
  - [ ] 对已跟踪的构建产物执行 `git rm --cached`（不删除本地文件）

- [ ] 库输出规范
  - [ ] 库内仅保留错误级输出，已清理（复查 `src/、include/`）
  - [ ] 为错误路径保留明确提示（模型缺失、RKNN 失败、维度异常）

- [ ] 交付自测流程
  - [ ] 本地执行打包脚本，解包后验证：`runtime/examples/pose_camera_bytetrack_homography` 可运行
  - [ ] `ldd` 检查 `libdetector_lib.so、librknnrt.so` 解析正常
  - [ ] 按 `README_QUICK.md` 步骤完成安装/运行/卸载回归

- [ ] Git 与发布
  - [ ] 完成上述变更后提交并推送（Gitee/GitHub）
  - [ ] 创建 Tag：`v1.0.4`
  - [ ] 将 `dist/*.tar.gz` 作为发行资产归档（可选）

备注：暂不删除任何文件与目录；完成分层与引用检索后，再执行删除旧结构的操作。


