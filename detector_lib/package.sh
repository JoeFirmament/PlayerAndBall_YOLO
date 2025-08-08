#!/bin/bash
set -e

echo "[DEPRECATED] package.sh 已废弃，请使用 ./create_distribution_package.sh"
if [ -x "./create_distribution_package.sh" ]; then
  exec ./create_distribution_package.sh "$@"
else
  echo "[ERROR] 未找到 create_distribution_package.sh，请从 detector_lib 目录运行或更新仓库。"
  exit 1
fi