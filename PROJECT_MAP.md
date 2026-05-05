# PROJECT_MAP

本文档是本仓库的项目地图，面向后续维护者快速理解 SLIMER 的代码组织、数据流和常用入口。

## 项目定位

SLIMER 是一个面向 zero-shot named entity recognition 的 LLM 微调与评估项目。核心思想是把每个实体类型转换成一条抽取指令，并可选地在指令中加入该实体类型的 Definition 和 Guidelines，让模型在见过更少实体类型的情况下泛化到未见实体类型。

项目主要覆盖三条链路：

1. 将 BIO、PileNER、UniNER conversation 格式数据转换为 SLIMER/GenQA 格式。
2. 用 PileNER 子集对 LLaMA 类模型做 LoRA supervised fine-tuning。
3. 用 vLLM 在 CrossNER、MIT、BUSTER 上评估 zero-shot NER。

## 总体数据流

```text
原始数据
  - data/CrossNER/ner_data/*/*.txt      BIO token/tag
  - data/MIT/*/*.txt                    MIT token/tag, 列顺序不同
  - data/eval_data_UniNER/test_data/*.json
  - Universal-NER/Pile-NER-type         Hugging Face dataset
  - ../dataset/Pile-NER-type            本地 PileNER conversation 文件
  - expertai/BUSTER                     Hugging Face dataset

        |
        v

src/data_handlers/
  - BIO/UniNER/PileNER 解析
  - 实体类型名称规范化
  - Definition & Guidelines 注入
  - 采样正负样本

        |
        v

SLIMER/GenQA 样本
  {
    "doc_tag_pairID": "...",
    "tagName": "...",
    "input": "...",
    "instruction": "...",
    "output": "[\"span_1\", \"span_2\"]"
  }

        |
        +--> src/SFT_finetuning/training/finetune_sft.py
        |      基于 ../model/Llama-3.1-8B-Instruct 训练 LoRA，输出 trained_models/*
        |
        +--> src/SFT_finetuning/commons/merge_lora_weights.py
        |      合并基础模型和 LoRA，输出 merged_models/*
        |
        +--> src/SFT_finetuning/evaluating/evaluate_vLLM.py
               vLLM 推理，UniNER 脚本评估，输出 predictions/*
```

## 顶层目录

```text
.
├── AGENTS.md
├── README.md
├── PROJECT_MAP.md
├── requirements.txt
├── LICENSE
├── data/
│   ├── CrossNER/
│   ├── MIT/
│   ├── eval_data_UniNER/
│   └── pileNER/
└── src/
    ├── data_handlers/
    └── SFT_finetuning/
```

后续实验在服务器上使用仓库同级目录中的本地基础模型目录，不属于本仓库，也不应提交到版本控制：

```text
../model/
└── Llama-3.1-8B-Instruct/
```

### 根目录文件

- `AGENTS.md`：面向自动化代理和维护者的项目级工作规则，包含成本控制、数据切分隔离、验证策略和文档同步要求。
- `README.md`：项目简介、基本目录结构、环境安装和训练/合并/评估常用命令。
- `PROJECT_MAP.md`：本项目地图，说明目录结构、数据流、关键入口和维护注意事项。
- `requirements.txt`：运行依赖，包括 `torch`、`transformers`、`peft`、`vllm`、`huggingface-hub`、`datasets`、`numpy`。
- `.gitignore`：当前只忽略 `.DS_Store`、`.idea`、`.env`、`/venv/`。
- `LICENSE`：Apache 2.0。

本项目没有 `setup.py` 或 `pyproject.toml`，脚本通常假设从仓库根目录运行，并设置 `PYTHONPATH` 为项目根目录。

## 外部模型目录

后续实验统一使用仓库同级目录下的本地 Llama 3.1 8B Instruct 模型作为唯一基础模型：

- `../model/Llama-3.1-8B-Instruct`：本地基础/指令模型权重目录。

该目录位于服务器上，本地开发环境可能不可见。从仓库根目录运行评估时，可直接把该路径作为 `evaluate_vLLM.py` 的 `merged_model_name` 参数，并配套使用 `LLaMA3-chat` 模板。Windows PowerShell 中可写作 `..\model\Llama-3.1-8B-Instruct`。

后续实验沿用项目原有输出路径：LoRA adapter 保存到 `trained_models/`，合并后的模型保存到 `merged_models/`。

当前训练配置和 LoRA 合并脚本保留历史 LLaMA 2 默认值；后续训练或合并实验应显式把 `base_model` 指向 `../model/Llama-3.1-8B-Instruct`，并使用 `LLaMA3-chat` 相关模板。输出路径继续沿用 `trained_models/` 和 `merged_models/`。

## 数据目录

### `data/CrossNER/`

CrossNER BIO 数据，包含：

- `ai`
- `conll2003`
- `literature`
- `music`
- `politics`
- `science`

路径形如 `data/CrossNER/ner_data/ai/train.txt`、`dev.txt`、`test.txt`。每行是 `token label`，空行分隔句子。

### `data/MIT/`

MIT Movie 和 MIT Restaurant BIO 数据：

- `data/MIT/movie/train.txt`
- `data/MIT/movie/test.txt`
- `data/MIT/restaurant/train.txt`
- `data/MIT/restaurant/test.txt`

MIT 文件的列顺序与通用 BIO 读取逻辑相反，`data_handler_MIT.py` 会读取后交换 `tokens` 和 `labels`。

### `data/eval_data_UniNER/`

来自 UniNER 的测试集 conversation 格式 JSON，用于评估时转换为 SLIMER 格式：

- `CrossNER_ai.json`
- `CrossNER_literature.json`
- `CrossNER_music.json`
- `CrossNER_politics.json`
- `CrossNER_science.json`
- `mit-movie.json`
- `mit-restaurant.json`

这类样本结构大致是：

```json
{
  "id": "CrossNER_AI_9",
  "conversations": [
    {"from": "human", "value": "Text: ..."},
    {"from": "gpt", "value": "I've read this text."},
    {"from": "human", "value": "What describes algorithm in the text?"},
    {"from": "gpt", "value": "[\"naive Bayes classifier\"]"}
  ]
}
```

### `data/pileNER/`

已生成的 PileNER 子集，采用 SLIMER/GenQA JSONL 格式。当前包含带 Definition & Guidelines 和不带 Definition & Guidelines 的 391 类实体子集，例如：

- `5pos_5neg_perNE_top391NEs_TrueDef/train.jsonl`
- `5pos_5neg_perNE_top391NEs_TrueDef/validation.jsonl`
- `5pos_5neg_perNE_top391NEs_TrueDef/test.jsonl`
- `5pos_5neg_perNE_top391NEs_FalseDef/*`

训练脚本也会按参数重新生成类似目录。

`build_pileNER_391_plain_genqa.py` 会从仓库同级的 `../dataset/Pile-NER-type` 本地目录读取 PileNER conversation 文件，输出无切分、无 `instruction` 字段的 391 类全集：

- `pileNER_391_all.jsonl`

该文件每行包含 `doc_tag_pairID`、`tagName`、`input`、`output`，其中 `output` 是 JSON list 字符串。

## `src/data_handlers/`

该目录负责所有数据读取、格式转换、实体类型映射和 D&G prompt 构造。

### `Data_Interface.py`

NER 数据集适配的抽象基类。新数据集想接入 SLIMER 时，重点实现：

- `load_datasetdict_BIO(path_to_BIO, test_only=False)`：读取原始数据并返回 `datasets.DatasetDict`，每条样本含 `id`、`tokens`、`labels`。
- `get_map_to_extended_NE_name()`：把短标签映射到自然语言实体名，例如 `PER -> PERSON`。

重要方法：

- `read_bio_file()`：读取 `token label` 格式 BIO 文件。
- `get_ne_categories()`：从 BIO 标签中收集实体类型。
- `extract_gold_spans_per_ne_category()`：把 BIO 标签恢复成 gold span 文本和字符位置。
- `load_DeG_per_NEs()`：读取每类实体的 Definition & Guidelines。
- `convert_dataset_for_SLIMER()`：把每个文档展开为 `document x entity_type` 的 SLIMER 样本。
- `get_Npos_Mneg_per_topXtags()`：按实体类型抽取固定数量正负样本。

### 数据集适配器

- `data_handler_CrossNER.py`：适配 CrossNER 的 `ai`、`literature`、`music`、`politics`、`science` 等子数据集，并把 `programlang` 等短名映射为自然语言名称。
- `data_handler_MIT.py`：适配 MIT Movie/Restaurant。注意当前读取函数会取前 100 条样本。
- `data_handler_BUSTER.py`：适配 Hugging Face 上的 `expertai/BUSTER`，评估时使用 `FOLD_5` 作为 test fold。
- `data_handler_pileNER.py`：PileNER 的专用处理器，不继承 `Data_Interface`。它从 `Universal-NER/Pile-NER-type` conversation 样本中抽取 context、question、answer，构建 MSEQA/GenQA 格式数据，过滤 MIT/CrossNER/BUSTER 重叠实体类型，并生成训练用 JSONL。
- `build_pileNER_391_plain_genqa.py`：从本地 `../dataset/Pile-NER-type` 的 `.json`、`.jsonl`、`.parquet` conversation 文件构建无切分 391 类 JSONL。该脚本只做 QA 抽取、实体类型筛选和规范化，不生成 Definition & Guidelines，也不输出 `instruction`。

### `questions/`

存放每个数据集的实体问题、Definition & Guidelines 和示例句：

```text
questions/
├── BUSTER/
├── MIT/
├── crossNER/
└── pileNER/
```

常见文件类型：

- `what_describes_questions/*.txt`：实体类型问题列表。
- `gpt_guidelines/*_NE_definitions.json`：实体类型的自然语言名称、Definition、Guidelines。
- `sentences_per_ne_type*.json`：为生成或校验 D&G 使用的实体示例句。

`path_to_DeG` 或 `path_to_NE_guidelines_json` 指向这些 JSON 后，转换逻辑会把 D&G 注入 `instruction` 字段。

## `src/SFT_finetuning/`

训练、推理、模板和通用模型工具集中在这里。

### `templates/`

prompt 模板目录：

- `LLaMA2-chat.json`：LLaMA 2 chat 外层 prompt。
- `LLaMA3-chat.json`：LLaMA 3 instruct 外层 prompt。
- `SLIMER_instruction_template.json`：SLIMER D&G 指令模板。
- `SLIMER3_instruction_template.json`：更严格要求只返回 JSON list 的 SLIMER 指令模板。

外层模板由 `Prompter` 使用；SLIMER 实体抽取指令模板由 `SLIMER_instruction_prompter` 使用。

### `commons/`

通用训练与推理工具：

- `prompter.py`：`Prompter` 负责把 `instruction + input + optional output` 放进 LLaMA 模板；`SLIMER_instruction_prompter` 负责生成实体抽取指令。
- `preprocessing.py`：tokenize、prompt 拼接、label mask、输入截断。训练时默认只在 response 部分计算 loss。
- `initialization.py`：加载 tokenizer/model、设置 LLaMA token id、加载 LoRA、包装 PEFT LoRA。
- `generation.py`：Transformers 原生单条和批量生成辅助函数。
- `merge_lora_weights.py`：将 LoRA adapter 合并回 base model。
- `llama_patch.py`：Flash Attention 相关 patch 代码，目前主流程中基本处于注释或可选状态。
- `basic_utils.py`：JSON、路径、日期字符串等小工具。

### `training/finetune_sft.py`

LoRA SFT 主入口。

脚本参数：

```text
python src/SFT_finetuning/training/finetune_sft.py [--with_guidelines] number_NEs number_pos_samples_per_NE number_neg_samples_per_NE
```

典型流程：

1. 调用 `data_handler_pileNER.build_dataset_MSEQA_format_with_n_samples_per_NE_pos_neg()` 从 PileNER 构建子集。
2. 调用 `convert_MSEQA_dataset_to_GenQA_format_SI()` 生成 `data/pileNER/{dataset_name}/train.jsonl` 和 `validation.jsonl`。
3. 读取 `training_config/llama2_4_NER_XDef_NsamplesPerNE.yml`。
4. 覆盖 `data_path`、`val_data_path`、`output_dir`。
5. 初始化模型与 LoRA，使用 `transformers.Trainer` 训练。
6. 保存 adapter 到 `trained_models/LLaMA2_7B_*`，并复制训练配置。后续实验可沿用该输出目录，但应按 Llama 3.1 更新命名。

README 中的示例：

```bash
PYTHONPATH=$(pwd) python src/SFT_finetuning/training/finetune_sft.py 391 5 5 --with_guidelines
```

PowerShell 可使用：

```powershell
$env:PYTHONPATH = (Get-Location).Path
python src/SFT_finetuning/training/finetune_sft.py 391 5 5 --with_guidelines
```

### `training_config/`

当前有一个历史 LLaMA 2 训练配置文件，后续实验应把它作为待更新模板而不是最终默认值：

- `llama2_4_NER_XDef_NsamplesPerNE.yml`

关键默认值：

- `base_model: meta-llama/Llama-2-7b-chat-hf`
- `prompt_template_name: LLaMA2-chat`
- `batch_size: 32`
- `micro_batch_size: 1`
- `num_epochs: 10`
- `cutoff_len: 768`
- `use_lora: True`
- `lora_target_modules: q_proj, v_proj, k_proj`

该历史 base model 需要 Hugging Face 账号拥有 LLaMA 2 访问权限。后续实验只使用本地 `../model/Llama-3.1-8B-Instruct` 作为基础模型，因此运行训练前应更新或新增训练配置，将 `base_model` 指向该本地路径，并把 `prompt_template_name` 设为 `LLaMA3-chat`。输出路径继续沿用 `trained_models/`。

### `evaluating/evaluate_vLLM.py`

zero-shot NER 评估主入口。它会依次评估：

- CrossNER：`ai`、`literature`、`music`、`politics`、`science`
- MIT：`movie`、`restaurant`
- BUSTER：`BUSTER`

脚本参数：

```text
python src/SFT_finetuning/evaluating/evaluate_vLLM.py merged_model_name template_name [--with_guidelines]
```

README 示例：

```bash
PYTHONPATH=$(pwd) python src/SFT_finetuning/evaluating/evaluate_vLLM.py ../model/Llama-3.1-8B-Instruct LLaMA3-chat --with_guidelines
PYTHONPATH=$(pwd) python src/SFT_finetuning/evaluating/evaluate_vLLM.py expertai/SLIMER LLaMA2-chat --with_guidelines
PYTHONPATH=$(pwd) python src/SFT_finetuning/evaluating/evaluate_vLLM.py expertai/SLIMER-LLaMA3 LLaMA3-chat --with_guidelines
```

评估过程：

1. 将 UniNER/MIT/CrossNER/BUSTER 测试数据转换成 SLIMER 格式。
2. 使用 `Prompter` 生成完整 chat prompt。
3. 对 CrossNER/MIT 输入做 token 截断。
4. 对 BUSTER 长文档做 sliding window chunk，再聚合 chunk 预测。
5. 使用 vLLM 生成 JSON list 风格预测。
6. 调用 `uniNER_official_eval_script.NEREvaluator` 计算 micro、macro、weighted 指标。
7. 保存预测到 `predictions/{model_name}/{subdataset}.json`。

### `evaluating/eval_utils.py`

- `chunk_document_with_sliding_window()`：BUSTER 长文档切片。
- `aggregate_preds_from_chunks()`：把 chunk 级 JSON list 预测聚合回文档级预测。

注意：函数文档写的是按 words 切片，但当前实现对字符串做字符区间切片。

### `evaluating/uniNER_official_eval_script.py`

UniNER 官方评估脚本副本。主要逻辑：

- 从模型输出中截取第一个 `[...]` JSON list。
- 标准化大小写、标点、冠词和空白。
- 使用集合式去重后的实体文本做 exact match。
- `partial_evaluate()` 支持基于词重叠 F1 的宽松匹配。

## 常用命令

安装依赖：

```bash
pip install --upgrade pip
pip install -r ./requirements.txt
```

评估本地 Llama 3.1 8B Instruct 模型：

```bash
PYTHONPATH=$(pwd) python src/SFT_finetuning/evaluating/evaluate_vLLM.py ../model/Llama-3.1-8B-Instruct LLaMA3-chat --with_guidelines
```

评估微调或合并后的本地模型：

```bash
PYTHONPATH=$(pwd) python src/SFT_finetuning/evaluating/evaluate_vLLM.py ./merged_models/<merged_model_name> LLaMA3-chat --with_guidelines
```

评估公开模型：

```bash
PYTHONPATH=$(pwd) python src/SFT_finetuning/evaluating/evaluate_vLLM.py expertai/SLIMER LLaMA2-chat --with_guidelines
```

训练 LoRA：

```bash
PYTHONPATH=$(pwd) python src/SFT_finetuning/training/finetune_sft.py 391 5 5 --with_guidelines
```

从本地 PileNER conversation 文件生成无切分 391 类 JSONL：

```powershell
$env:PYTHONPATH = (Get-Location).Path
python src/data_handlers/build_pileNER_391_plain_genqa.py --overwrite
```

smoke test 可限制原始样本数并写到临时输出：

```powershell
python src/data_handlers/build_pileNER_391_plain_genqa.py --max_raw_samples 20 --output_path data/pileNER/pileNER_391_all.smoke.jsonl --overwrite
```

合并 LoRA：

```bash
PYTHONPATH=$(pwd) python src/SFT_finetuning/commons/merge_lora_weights.py 391 5 5 --with_guidelines
```

评估合并后的模型：

```bash
PYTHONPATH=$(pwd) python src/SFT_finetuning/evaluating/evaluate_vLLM.py LLaMA2_7B_5pos_5neg_perNE_top391NEs_TrueDef LLaMA2-chat --with_guidelines
```

## 关键数据结构

### BIO 样本

`Data_Interface` 期望 BIO 数据转换后为：

```python
{
    "id": "ai:train:0",
    "tokens": ["Popular", "approaches", "..."],
    "labels": ["O", "O", "B-product", "..."]
}
```

### SLIMER/GenQA 样本

训练和评估的核心样本格式：

```python
{
    "doc_tag_pairID": "CrossNER_AI_9:0",
    "tagName": "algorithm",
    "input": "Typical generative model approaches include ...",
    "instruction": "Extract the Named Entities of type ALGORITHM ...",
    "output": "[\"naive Bayes classifier\", \"Gaussian mixture model\"]"
}
```

`output` 是 JSON dump 后的字符串，不是 Python list。

### Definition & Guidelines JSON

D&G 文件通常按实体类型存储：

```python
{
    "algorithm": {
        "named_entity": "algorithm",
        "real_name": "algorithm",
        "gpt_answer": "{\"Definition\": \"...\", \"Guidelines\": \"...\"}"
    }
}
```

代码会把 `gpt_answer` 解析成含 `Definition` 和 `Guidelines` 的字典，再写入 prompt。

## 新增数据集的接入步骤

1. 把数据整理成 BIO 或在自定义 handler 中读成 `DatasetDict`。
2. 新增 `src/data_handlers/data_handler_<DATASET>.py`，继承 `Data_Interface`。
3. 实现 `load_datasetdict_BIO()` 和 `get_map_to_extended_NE_name()`。
4. 在 `src/data_handlers/questions/<DATASET>/` 下放入实体类型问题和 D&G JSON。
5. 实例化 handler 时传入：

```python
handler = MyDataset(
    path_to_BIO="...",
    path_to_templates="./src/SFT_finetuning/templates",
    SLIMER_prompter_name="SLIMER_instruction_template",
    path_to_DeG="./src/data_handlers/questions/<DATASET>/gpt_guidelines/<DATASET>_NE_definitions.json",
)
```

6. 使用 `handler.dataset_dict_SLIMER` 作为训练或评估输入。

## 生成目录与运行产物

这些目录由脚本生成或期望存在：

- `trained_models/`：LoRA adapter。
- `merged_models/`：base model + LoRA 合并后的模型。
- `predictions/`：评估脚本保存的预测 JSON。
- `saved_models/`：部分 helper 的默认示例路径。
- `data/pileNER/<dataset_name>/`：训练脚本按参数生成的 JSONL 子集。
- `data/pileNER/pileNER_391_all.jsonl`：本地 PileNER conversation 文件转换出的无切分 391 类全集，不包含 `instruction` 字段。

`trained_models/` 和 `merged_models/` 是项目既有输出目录；后续实验继续沿用这两个路径。

## 维护注意事项

- 根目录 `AGENTS.md` 是本仓库的代理协作规则。每次修改项目后，都必须同步检查 `README.md` 和 `PROJECT_MAP.md`：若修改影响目录结构、运行命令、依赖、数据流、实验假设、生成产物或用户可见行为，应同时更新这两个文档；若无需改动，也应在交付说明中说明已经检查。
- 脚本中的路径大多是相对仓库根目录的相对路径，建议始终从项目根目录运行。
- `data_handler_pileNER.extract_context_quests_answers()` 会对传入的 `conversation` 使用 `pop()`，如果调用方后续还要复用原始 conversation，应先传入副本。
- `Data_Interface.load_DeG_per_NEs()` 和 PileNER 转换逻辑会对 `gpt_answer` 使用 `eval()`，D&G JSON 应来自可信来源。
- 评估脚本默认 `cutoff_len=768`、`max_new_tokens=128`，BUSTER 走 chunk 聚合，其它数据集走 token 截断。
- README 同时给出类 Unix shell 和 Windows PowerShell 的 `PYTHONPATH` 设置方式。
