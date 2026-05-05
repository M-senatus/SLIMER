# SLIMER

SLIMER 是一个面向 **zero-shot NER** 的 LLM 研究实验项目。项目主要用于将 NER 数据转换为生成式问答格式，并基于 Definition & Guidelines prompt 进行 LoRA 微调、模型合并和 vLLM 评估。

相关链接：

- Paper: [Show Less, Instruct More: Enriching Prompts with Definitions and Guidelines for Zero-Shot NER](https://arxiv.org/abs/2407.01272)
- Model: [expertai/SLIMER](https://huggingface.co/expertai/SLIMER)
- Model: [expertai/SLIMER-LLaMA3](https://huggingface.co/expertai/SLIMER-LLaMA3)
- 后续实验唯一基础模型：`../model/Llama-3.1-8B-Instruct`

## 项目结构

```text
.
|-- data/
|   |-- CrossNER/
|   |-- MIT/
|   |-- eval_data_UniNER/
|   `-- pileNER/
|-- src/
|   |-- data_handlers/
|   |-- SFT_finetuning/
|   |   |-- commons/
|   |   |-- evaluating/
|   |   |-- templates/
|   |   |-- training/
|   |   `-- training_config/
|-- requirements.txt
|-- PROJECT_MAP.md
`-- README.md
```

后续实验在服务器上使用仓库同级目录中的本地基础模型目录，不属于本仓库：

```text
../model/
`-- Llama-3.1-8B-Instruct/
```

目录说明：

- `data/`：CrossNER、MIT、UniNER 测试数据，以及生成的 PileNER 子集。
- `src/data_handlers/`：数据读取、格式转换、实体类型映射和 Definition & Guidelines 注入。
- `src/data_handlers/questions/`：各数据集的实体问题和 Definition & Guidelines 文件。
- `src/data_handlers/hierarchical_guideline_prompt.py`：层次化标注规范 prompt 构造与 JSON schema 校验。
- `src/SFT_finetuning/templates/`：LLaMA chat 模板和 SLIMER 指令模板。
- `src/SFT_finetuning/training/`：LoRA SFT 训练入口。
- `src/SFT_finetuning/commons/`：prompt、preprocessing、模型初始化、LoRA 合并等公共逻辑。
- `src/SFT_finetuning/evaluating/`：vLLM 推理和评估脚本。

更详细的文件说明见 [PROJECT_MAP.md](PROJECT_MAP.md)。

## 环境安装

从仓库根目录安装依赖：

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell：

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

运行脚本前设置 `PYTHONPATH`。

Bash / Linux：

```bash
export PYTHONPATH=$(pwd)
```

Windows PowerShell：

```powershell
$env:PYTHONPATH = (Get-Location).Path
```

## 本地模型约定

后续实验统一使用仓库同级目录下的本地 Llama 3.1 8B Instruct 模型作为唯一基础模型：

```text
../model/Llama-3.1-8B-Instruct
```

该目录位于服务器上的仓库同级目录，本地开发环境可能不可见。从仓库根目录运行时，vLLM 评估可直接把该相对路径作为模型参数，并配套使用 `LLaMA3-chat` 模板。

后续实验沿用项目原有输出路径：LoRA adapter 保存到 `trained_models/`，合并后的模型保存到 `merged_models/`。

## 使用方法

评估唯一基础模型：

```bash
PYTHONPATH=$(pwd) python src/SFT_finetuning/evaluating/evaluate_vLLM.py ../model/Llama-3.1-8B-Instruct LLaMA3-chat --with_guidelines
```

评估公开 LLaMA 2 模型：

```bash
PYTHONPATH=$(pwd) python src/SFT_finetuning/evaluating/evaluate_vLLM.py expertai/SLIMER LLaMA2-chat --with_guidelines
```

评估公开 LLaMA 3 模型：

```bash
PYTHONPATH=$(pwd) python src/SFT_finetuning/evaluating/evaluate_vLLM.py expertai/SLIMER-LLaMA3 LLaMA3-chat --with_guidelines
```

训练 LoRA（运行前先按“注意事项”同步训练配置中的基础模型和模板）：

```bash
PYTHONPATH=$(pwd) python src/SFT_finetuning/training/finetune_sft.py 391 5 5 --with_guidelines
```

从本地 PileNER conversation 文件生成无切分 391 类 JSONL：

```powershell
$env:PYTHONPATH = (Get-Location).Path
python src/data_handlers/build_pileNER_391_plain_genqa.py --overwrite
```

构造层次化标注规范 prompt：

```python
from src.data_handlers.hierarchical_guideline_prompt import build_hierarchical_guideline_messages
```

交互式批量生成可使用 `src/data_handlers/prompt_gpt_for_guidelines.ipynb`。该 notebook 当前输出六字段 JSON：`entity_type`、`definition`、`include_rules`、`exclude_rules`、`boundary_rules`、`confusable_type_rules`。

smoke test 可限制原始样本数并写到临时输出：

```powershell
python src/data_handlers/build_pileNER_391_plain_genqa.py --max_raw_samples 20 --output_path data/pileNER/pileNER_391_all.smoke.jsonl --overwrite
```

合并 LoRA（运行前先按“注意事项”同步合并脚本中的基础模型）：

```bash
PYTHONPATH=$(pwd) python src/SFT_finetuning/commons/merge_lora_weights.py 391 5 5 --with_guidelines
```

评估微调或合并后的本地模型：

```bash
PYTHONPATH=$(pwd) python src/SFT_finetuning/evaluating/evaluate_vLLM.py ./merged_models/<merged_model_name> LLaMA3-chat --with_guidelines
```

Windows PowerShell 中可先设置 `PYTHONPATH`，再去掉命令前的 `PYTHONPATH=$(pwd)`：

```powershell
$env:PYTHONPATH = (Get-Location).Path
python src/SFT_finetuning/evaluating/evaluate_vLLM.py ..\model\Llama-3.1-8B-Instruct LLaMA3-chat --with_guidelines
```

## 数据与输出

主要输入数据与模型：

- `data/CrossNER/ner_data/`
- `data/MIT/`
- `data/eval_data_UniNER/test_data/`
- `data/pileNER/`
- `../dataset/Pile-NER-type`：本地 PileNER conversation 文件目录，供 `build_pileNER_391_plain_genqa.py` 读取。
- `../model/Llama-3.1-8B-Instruct`：服务器上仓库外的本地 Llama 3.1 8B Instruct 基础模型目录，作为后续实验唯一基础模型来源。

常见输出目录：

- `data/pileNER/{dataset_name}/`：训练脚本生成的 JSONL 数据。
- `data/pileNER/pileNER_391_all.jsonl`：无切分 391 类 PileNER JSONL，字段为 `doc_tag_pairID`、`tagName`、`input`、`output`，不包含 `instruction`；生成时会跳过含非 ASCII 字符的样本，保证输出只含英文类字符。
- `definitions/*_hierarchical_guidelines.json`：`prompt_gpt_for_guidelines.ipynb` 可生成的层次化标注规范 JSON，属于 GPT 生成的实验产物。
- 所有 `*.jsonl` 文件视为生成数据或实验产物，默认不纳入 git 跟踪；需要共享时应通过外部数据存储或明确约定的发布流程处理。
- `trained_models/`：LoRA adapter。
- `merged_models/`：合并后的模型。
- `predictions/`：评估预测结果。

## 注意事项

- 后续训练、合并和推理都以 `../model/Llama-3.1-8B-Instruct` 作为唯一基础模型，并使用 `LLaMA3-chat` 模板。
- 当前训练配置和部分脚本名称仍保留历史 LLaMA 2 痕迹；运行 LoRA 训练或合并前，应先将对应配置或脚本中的 `base_model` 指向 `../model/Llama-3.1-8B-Instruct`，并将模板设为 `LLaMA3-chat`。输出路径继续沿用 `trained_models/` 和 `merged_models/`。
- vLLM 推理通常需要 Linux + CUDA GPU 环境。
- 训练、合并和完整评估可能消耗较多 GPU 时间和磁盘空间。
- 原始数据建议只读，转换后的数据写入单独生成目录。

## License

本仓库使用 Apache License 2.0。详见 [LICENSE](LICENSE)。
