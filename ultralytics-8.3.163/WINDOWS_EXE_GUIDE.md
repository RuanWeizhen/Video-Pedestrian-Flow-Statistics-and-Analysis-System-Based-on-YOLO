# Windows 可执行程序打包说明

## 项目类型判断

当前项目是 Python + PyQt5 桌面程序，核心入口在 `run_gui.py`，GUI 包入口在 `pedestrian_system/gui/`.

## 打包方案

已采用 PyInstaller 方案，生成方式为 `--onefile`（单 EXE 文件）+ `--windowed`（无控制台窗口）。
模型文件（.pt）、配置文件（.yaml）、视频资源**禁止打包进 EXE 内部**，所有外部资源存放于 EXE 所在同级目录，支持用户随时替换更新而无需重新打包。

## 发布目录结构

打包完成后，发布目录应组织如下：

```text
行人检测系统/
  行人检测系统.exe           ← 双击启动
  models/                     ← 存放模型文件（.pt / .onnx）
    best.pt                   ← 主检测模型
    yolov8n-face.pt           ← 人脸模糊模型（可选）
  config/                     ← 存放配置文件（.yaml / .yml）
    pedestrian_demo.yaml      ← 默认配置
  videos/                     ← 存放测试视频（可选）
    test.mp4
```

## 代码资源加载规则

程序中实现了三套路径解析函数（位于 `pedestrian_system/utils/paths.py`），适配开发环境与打包 EXE 环境：

| 函数 | 用途 | 开发环境 | EXE 环境 |
|------|------|----------|----------|
| `external_resource_path()` | 模型/配置/视频等外部资源 | 从 `pedestrian_system/` 目录查找 | 从 **EXE 同级目录**查找（禁止从 `_MEIPASS` 临时目录读取） |
| `resource_path()` | 内嵌资源（如 seed 数据库） | 从 `pedestrian_system/` 目录查找 | 从 `_MEIPASS` 临时目录查找 |
| `writable_path()` | 可写数据（日志/输出/数据库） | 从 `pedestrian_system/` 目录写入 | 写入 **用户文档目录** `Documents/行人检测系统/` |

- **环境智能识别**：自动判断当前运行环境（`sys.frozen` / `sys._MEIPASS` / `sys.executable`）
- **路径不存在时**：`external_resource_path()` 会输出警告日志并返回"最佳猜测"路径，便于调试
- **所有路径分隔符**：统一使用 `pathlib.Path`，兼容 Windows 路径格式
- **日志输出**：每个路径函数在 INFO 级别记录实际解析路径，WARNING 级别记录缺失路径

## 一键打包

先进入 `ultralytics-8.3.163` 目录，然后运行：

```bat
build_exe.bat
```

或直接运行：

```bat
python build_exe.py
```

打包产物默认输出到 `dist/行人检测系统.exe`。

`.spec` 配置文件中已明确排除所有 `.pt`、`.pth`、`.yaml`、`.yml` 后缀的资源文件，并过滤了 `models/`、`config/`、`videos/`、`runs/` 等目录，确保外部资源不会进入 EXE。

## 首次启动注意事项

1. 将 `dist/行人检测系统.exe` 复制到目标部署目录
2. 在 EXE 同级创建 `models/`、`config/`、`videos/` 文件夹
3. 将模型文件（如 `best.pt`）放入 `models/`
4. 将配置文件（如 `pedestrian_demo.yaml`）放入 `config/`，并修改其中的 `model_path` 指向 `models/best.pt`
5. 将测试视频放入 `videos/`
6. 确保部署目录有写权限（或程序会自动将数据写入 `Documents/行人检测系统/`）

### 模型替换

如需更换模型，将新 `.pt` 文件放入 `models/` 目录，修改 `config/` 中的配置文件 `detector.model_path` 即可，无需重新打包。

## 常见问题

### 1. 提示找不到模型文件
检查 `models/` 目录下是否存在对应的 `.pt` 文件，配置文件中的 `model_path` 是否正确（建议使用相对路径如 `models/best.pt`）。

### 2. 提示找不到配置文件
检查 `config/` 目录下是否存在配置文件，确认文件名和路径正确。

### 3. 数据库无法写入
程序会自动将数据库等可写数据写入 `Documents/行人检测系统/outputs/`，确保用户文档目录可写。

### 4. 窗口打开后闪退
查看 `Documents/行人检测系统/logs/startup.log` 了解闪退原因，常见原因是模型文件缺失或路径配置错误。

### 5. Qt 平台插件问题
如果出现 `qt platform plugin` 相关报错，通常是 PyQt5 的插件目录没有被正确打包，需要检查 PyInstaller 版本和 PyQt5 安装是否完整。

### 6. DLL 加载失败
检查是否缺少 VC++ 运行时库。对于 Torch GPU 版本，确保目标机器安装了合适的 CUDA 版本。

## 兼容性测试注意事项

- 目标机器**无需安装 Python**
- 目标机器**无需安装项目依赖**
- 如果在干净 Windows 虚拟机中测试，请确保已安装 VC++ Redistributable（Visual C++ 2015-2022）
- 替换模型文件后，重启 EXE 即可加载新模型
