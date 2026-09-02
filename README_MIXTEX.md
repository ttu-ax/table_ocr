# MixTeX 表格识别实验版

双击 `run_mixtex_gui.bat` 启动。界面和原来的 `table_recognizer_gui.py` 相同，但“API 地址”栏现在表示本地 MixTeX `onnx` 模型目录，默认值无需修改。

每次识别还会在原图片旁生成一个 `*_mixtex_raw.tex` 文件，便于检查模型的原始 LaTeX 输出。MixTeX 官方将表格识别描述为适用于清晰打印字体和较简单表格；对几十行、十几列、拍照透视明显的档案表，效果可能不如专用表格结构模型。

