# AGENTS.md

## 适用范围

本文件适用于整个 SLIMER 仓库。除非子目录中存在更具体的 `AGENTS.md`，后续所有自动化代理和维护者都应遵守这里的规则。

SLIMER 是一个 deep learning / LLM research experiment 项目，不是产品软件项目。典型任务包括：

- zero-shot NER 数据转换、训练、推理和评估
- Definition & Guidelines prompt 实验
- LoRA supervised fine-tuning
- vLLM / Transformers 推理
- 指标评估、消融、错误分析和复现实验

优先级固定为：

1. 正确性
2. 可复现性
3. 成本控制
4. 最小、可审计的改动

## 必须同步的文档

每次对项目做出修改后，都必须同步检查并对齐：

- `PROJECT_MAP.md`
- `README.md`

硬性要求：

- 如果修改影响目录结构、入口脚本、运行命令、依赖、数据流、实验假设、生成产物、配置含义或用户可见行为，必须同时更新 `PROJECT_MAP.md` 和 `README.md`。
- 如果修改只影响内部实现，也必须检查这两个文件是否仍准确；若无需改动，在最终回复中说明已经检查且无需更新。
- 新增、删除、重命名文件或目录时，优先更新 `PROJECT_MAP.md` 的结构说明。
- 改变安装、训练、合并、推理、评估或示例用法时，优先更新 `README.md` 的用户入口说明。
- 不要只改代码而留下过期文档。

## 工作流程

对任何非微小任务，按以下顺序执行：

1. 先阅读相关文件，理解当前实现。
2. 简短说明目标、影响文件、关键假设和验证方式。
3. 制定小范围计划。
4. 做最小必要修改。
5. 运行最小有意义的验证。
6. 汇报修改文件、验证命令、验证结果、剩余风险。

工作边界：

- 不要重构无关代码。
- 不要无故重命名文件、目录、函数、类、配置字段或实验目录。
- 不要静默改变实验假设、数据切分、标签定义、指标定义或 prompt 语义。
- 遵循仓库既有风格，优先清晰、显式、可复查的实现。

## 成本与安全约束

- 不要默认启动 full-scale training。
- 不要默认启动大规模真实 API batch。
- 不要默认下载大型模型、数据集或生成大量中间文件。
- 不要删除 checkpoints、logs、outputs、predictions、trained_models、merged_models、datasets 或原始数据，除非用户明确要求。
- 不要覆盖 raw data；转换数据应写到单独的生成目录。
- 不要修改 secrets、tokens、`.env`、SSH 设置或账号凭据，除非用户明确要求。

如果任务可能消耗明显 GPU 时间、API 预算、磁盘空间或网络流量，先给出：

1. 将要运行的命令
2. 预期规模
3. 成本和风险
4. tiny-run / smoke-test 验证路径

## 项目结构约定

当前主要结构如下：

- `src/data_handlers/`：数据读取、BIO/UniNER/PileNER 转换、实体类型映射、Definition & Guidelines 注入。
- `src/data_handlers/questions/`：各数据集的实体问题、Definition & Guidelines、示例句。
- `src/SFT_finetuning/templates/`：LLaMA chat 模板和 SLIMER 指令模板。
- `src/SFT_finetuning/commons/`：prompt、preprocessing、model 初始化、生成、LoRA 合并等公共逻辑。
- `src/SFT_finetuning/training/`：LoRA SFT 训练入口。
- `src/SFT_finetuning/training_config/`：训练配置。
- `src/SFT_finetuning/evaluating/`：vLLM 推理、chunk 聚合、UniNER 指标评估。
- `data/`：仓库内数据与生成的 PileNER 子集；原始数据视为只读。

脚本通常假设从仓库根目录运行，并设置 `PYTHONPATH` 为项目根目录。

PowerShell 示例：

```powershell
$env:PYTHONPATH = (Get-Location).Path
python src/SFT_finetuning/evaluating/evaluate_vLLM.py expertai/SLIMER LLaMA2-chat --with_guidelines
```

## 数据切分隔离

`train`、`dev` / `validation`、`test` 必须严格隔离。任何涉及数据处理、训练、验证、测试、缓存、prompt 构造、检索、特征工程、预处理、分析或实验编排的代码，都不得为了方便而破坏切分边界。

允许范围：

- `train` 只用于模型训练。
- `dev` / `validation` 只用于验证、调参、early stopping、模型选择和中间分析。
- 训练和验证流程可以读取 `train` / `dev` 的文本和标签。
- `test_infer` 只能读取原始 `test` 文本并生成预测，不能读取任何 `test` 标签或依赖标签算指标。
- `final_eval` 是唯一允许读取 `test` 标签并计算最终 Precision / Recall / F1 的阶段。

禁止范围：

- 非最终测试阶段不得读取 `test` 文本或标签。
- 不得用 `test` 构建 vocabulary、label set、实体类型描述、prompt 模板、示例、规则、缓存、索引、向量库或 prior knowledge。
- 不得在 logs、debug output、cache、pickle、npy、pt、json、faiss 或中间产物中泄露 `test` 标签。
- 不得在程序启动时默认一次性加载所有 splits。

代码级要求：

- 新增或修改 split loader 时，应显式区分训练、验证、test text-only inference 和 final evaluation。
- 任何可能访问 `test` 的流程都必须有明确 stage，例如 `train`、`dev`、`test_infer`、`final_eval`。
- `test_infer` 试图读取标签、非 `final_eval` 试图读取完整 test、非测试阶段试图访问 test 时，必须立即抛异常并停止。
- 保护逻辑必须写在代码里，不能只写注释或文档。

## LLM API、推理与训练规则

LLM API 实验：

- provider、model、base_url、temperature、max tokens、timeout、retry、batch size 等参数应可配置。
- 昂贵调用应支持 `max_samples`、缓存、断点续跑或 dry-run。
- 输出应保存到文件，不只打印到 stdout。
- prompt 模板应尽量与执行逻辑分离。

推理：

- 必须支持 tiny-run 或 small-sample 模式。
- input path、output path、batch size、device、dtype、max_samples 应尽量显式可控。
- 预测结果应保存到结构化目录，例如 `predictions/{model_name}/...`。
- 记录模型名、checkpoint、模板、batch size、precision、device、运行时间或吞吐量等关键信息。

fine-tuning：

- 训练前检查数据 schema、配置路径和模型权限。
- 明确区分 dry-run、smoke test、tiny-subset training 和 full training。
- 修改训练脚本后，先给出最小验证命令，不要自动继续完整训练。
- 每次训练应保存或记录确切 config。

评估和消融：

- 指标代码必须清楚说明输入假设。
- 比较实验时说明确切 config 差异。
- 错误分析样本应保存到文件，方便复查。
- 不要改变指标定义，除非用户明确要求。

## 验证策略

默认运行最小有意义的验证：

- Python 语法检查：`python -m py_compile <changed files>`
- import / smoke test
- tiny inference run
- tiny training step 或 tiny subset run
- evaluation smoke test

不要默认运行昂贵的 end-to-end 工作负载。若无法验证，必须明确说明原因。

## 编码风格

- 保持现有代码风格。
- 优先明确、简单、可读；避免不必要抽象。
- 新依赖必须有明确理由，并同步更新安装文档。
- 重要函数、脚本入口、复杂数据边界应有简短 docstring 或注释。
- 注释解释目的、输入输出、非显然逻辑、约束和边界；不要逐行复述代码。
- 修改逻辑时，同步更新相关注释，避免注释与实现不一致。

## 完成标准

任务只有在以下条件满足时才算完成：

- 请求的修改已经实现。
- `PROJECT_MAP.md` 和 `README.md` 已同步检查并按需更新。
- 已列出修改文件。
- 已运行最小相关验证，或说明无法运行的原因。
- 已说明主要假设、风险或未决问题。
