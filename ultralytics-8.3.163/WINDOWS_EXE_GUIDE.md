# Windows 可执行程序打包说明

## 项目类型判断

当前项目是 Python + PyQt5 桌面程序，核心入口在 `pedestrian_system/main.py`，界面入口在 `pedestrian_system/gui/__init__.py`。它不是 Electron、Tauri、Vue、React、Flask 或 FastAPI 项目。

## 打包方案

已采用 PyInstaller 方案，生成方式为 `onedir` 发布目录。这样可以把大模型、配置、视频样例、数据库文件放在 exe 旁边，不把它们塞进单个大 exe。

### 发布目录结构

打包完成后，建议目录类似下面这样：

```text
客流统计系统/
  客流统计系统.exe
  config/
    pedestrian_demo.yaml
    pedestrian_gui_saved.yaml
    test.yaml
  videos/
    test.mp4
    ...
  runs/
    train/
      yolov8m_ema_p3_896_sgd_cloud/
        weights/
          best.pt
  traffic.db
  outputs/
    traffic.db
```

## 一键打包

先进入 `ultralytics-8.3.163` 目录，然后运行：

```bat
build_exe.bat
```

或直接运行：

```bat
python build_exe.py
```

打包产物默认输出到 `dist/客流统计系统/`。

## 资源加载规则

程序已经改成优先从打包资源和运行目录读取文件。

- 配置文件默认读取 `config/pedestrian_demo.yaml`
- 默认检测模型读取 `runs/train/yolov8m_ema_p3_896_sgd_cloud/weights/best.pt`
- 示例视频读取 `videos/test.mp4`
- `traffic.db` 是随包带上的种子数据库，首次启动时会复制到 `outputs/traffic.db`
- 运行时数据库写入 `outputs/traffic.db`
- 用户登录数据库写入 `outputs/users.db`

## 运行说明

1. 双击 `客流统计系统.exe`
2. 先登录，再进入主界面
3. 选择视频或摄像头源
4. 点击开始处理

## 首次启动注意事项

- 确保 `best.pt` 仍然放在发布目录里的对应位置
- 如果你替换了模型文件，只要保持配置里的相对路径不变即可
- 如果要启用人脸模糊功能，还需要额外放入 `models/yolov8n-face.pt`
- 如果当前目录没有写权限，请把整个发布目录放到可写位置，比如桌面或文档目录

## 常见问题

### 1. 提示找不到模型文件

检查 `runs/train/yolov8m_ema_p3_896_sgd_cloud/weights/best.pt` 是否存在。

### 2. 提示找不到配置文件

检查 `config/pedestrian_demo.yaml` 是否存在。

### 3. 数据库无法写入

检查 `outputs/` 目录是否可写，或者是否被 Excel / WPS 占用。

### 4. 窗口打开后闪退

先在命令行运行一次 `python build_exe.py` 查看 PyInstaller 日志，常见原因是模型文件缺失或依赖没有被打进包。

### 5. Qt 平台插件问题

如果出现 `qt platform plugin` 相关报错，通常是 PyQt5 的插件目录没有被正确打包，需要检查 PyInstaller 版本和 PyQt5 安装是否完整。

## 额外说明

当前仓库里没有找到独立的 `.ico` 图标文件，所以打包脚本默认不指定图标。你后续如果补一个图标文件，可以在 spec 里加上 `icon=`。