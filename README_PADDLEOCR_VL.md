# PaddleOCR-VL-1.6 表格识别（vLLM 远程服务）实验版

本实验版把 `table_recognizer_gui.py` 的表头识别服务替换为服务器上的另一个 OCR 服务：
`192.168.10.26:8003` 上由 **vLLM** 启动的 **PaddleOCR-VL-1.6** 模型（OpenAI 兼容接口）。

## 启动

双击 `run_paddleocr_vl_gui.bat`，或运行：

```bat
python table_recognizer_gui_paddleocr_vl.py
```

界面与原生 `table_recognizer_gui.py` 完全相同；唯一的区别是它调用 `:8003` 的
PaddleOCR-VL-1.6 而不是 `:8087` 的表格识别服务。其它所有功能（多图合并/分离导出、
表头着色、冻结窗格、合并单元格、数字转 Excel 数值等）复用原代码，无需改动。

## CLI 批量测试

不打开界面也能直接测识别效果：

```bat
python test_paddle_ocr_vl.py --save " (13).jpg" " (14).jpg" " (15).jpg"
python test_paddle_ocr_vl.py --dir ./某目录
```

`--save` 会在每张图旁生成 `*_otsl.txt`（PaddleOCR-VL 的原始输出）和 `*_html.txt`
（转换成 HTML 的表格），便于核对模型输出；最后把所有结果写入 `paddleocr_vl_test.xlsx`。

## 模型接口契约（已适配）

* **Endpoint**：`POST http://192.168.10.26:8003/v1/chat/completions`
* **模型名**：`PaddleOCR-VL-1.6`
* **提示词**：`Table Recognition:`（触发 PaddleOCR-VL 的表格结构输出）
* **输出**：PaddleOCR-VL 特有的 OTSL 标记：
  - `<fcel>` 有文字的单元格、`<ecel>` 空单元格
  - `<nl>` 换行（一行结束）
  - `<lcel>` / `<ucel>` / `<xcel>` 表示向右/向下/跨行列的合并单元格
* **像素预算**：官方流水线对表格块使用 min 112896 / max 1003520 像素
  （约 1000×1000），本适配器会把矫正后的长边压到不超过 2000，避免超 Token 上限。
* **输出解析**：OTSL → HTML（`paddle_ocr_vl_adapter.convert_otsl_to_html`）→ 复用
  `table_recognizer_gui.parse_html_tables` 转成 `ParsedTable`，因此下游链路原样复用。

## 实测效果

对 `(10).jpg`（28×14 档案表）、`(13)/(14).jpg`（数字填充表）、`(15).jpg`
（含中文表头 + 手写“5月”的档案表）进行了测试：

* 行列结构识别准确（行数、列数正确，整数/汇总行数值核对无误）；
* 中文表头（日期、档号、案卷名称、…）识别正确，手写“5月”也能读出；
* 数字丢失较少，且末尾的“月总计”汇总行能完整还原；
* 单张 4284×5712 大图约 6–9 秒（取决于行数）。

### 档案表列对齐处理

这些档案表是固定的 14 列模板。PaddleOCR-VL 会把写在“日期/档号”表头处的**手写月份**
（如 `1月`/`2月`/`5月`）单独识别成一列（紧跟档号之后），导致每张图片的列数不一致
（15 / 13 / 12 列），并且有时会丢掉末尾的“挂接条目/装盒”两列。

`paddle_ocr_vl_adapter.normalize_archive_table` 会在识别后自动：

1. 依据表头关键词（档号/装订/编制页/…）判定是否为本档案表；
2. **删掉紧跟在档号后的“手写月份”这一多余列**；
3. 把所有图片统一重排到 `ARCHIVE_TABLE_HEADERS` 的 **14 列**标准网格上
   （缺失的末尾列补空），并强制表头为规范表头。

这样无论图片来自哪一页，生成的 Sheet 列数、列顺序、表头都完全一致，便于跨图片对齐汇总。

注意：该模型输出 OTSL 而不输出经典接口的 `overall_ocr_res` 坐标框，因此
`rebuild_grid_table`（依赖坐标做网格重建的增强逻辑）不会走；如果模型个别单元格
漏识别，建议用 `:8087` 或 PaddleOCR-VL 两种服务分别跑一遍对比。

## 文件说明

* `paddle_ocr_vl_adapter.py` —— 核心适配器：请求构造、OTSL→HTML 解析、图片像素预算。
* `table_recognizer_gui_paddleocr_vl.py` —— GUI 入口（替换 `recognize_image`）。
* `run_paddleocr_vl_gui.bat` —— 一键启动脚本。
* `test_paddle_ocr_vl.py` —— CLI 批量测试脚本。
