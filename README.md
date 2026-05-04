# SLIMER

SLIMER 是一个面向 **zero-shot NER** 的 LLM 研究实验项目。项目主要用于将 NER 数据转换为生成式问答格式，并基于 Definition & Guidelines prompt 进行 LoRA 微调、模型合并和 vLLM 评估。

相关链接：

- Paper: [Show Less, Instruct More: Enriching Prompts with Definitions and Guidelines for Zero-Shot NER](https://arxiv.org/abs/2407.01272)
- Model: [expertai/SLIMER](https://huggingface.co/expertai/SLIMER)
- Model: [expertai/SLIMER-LLaMA3](https://huggingface.co/expertai/SLIMER-LLaMA3)

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

目录说明：

- `data/`：CrossNER、MIT、UniNER 测试数据，以及生成的 PileNER 子集。
- `src/data_handlers/`：数据读取、格式转换、实体类型映射和 Definition & Guidelines 注入。
- `src/data_handlers/questions/`：各数据集的实体问题和 Definition & Guidelines 文件。
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

## 使用方法

评估公开 LLaMA 2 模型：

```bash
PYTHONPATH=$(pwd) python src/SFT_finetuning/evaluating/evaluate_vLLM.py expertai/SLIMER LLaMA2-chat --with_guidelines
```

评估公开 LLaMA 3 模型：

```bash
PYTHONPATH=$(pwd) python src/SFT_finetuning/evaluating/evaluate_vLLM.py expertai/SLIMER-LLaMA3 LLaMA3-chat --with_guidelines
```

训练 LoRA：

```bash
PYTHONPATH=$(pwd) python src/SFT_finetuning/training/finetune_sft.py 391 5 5 --with_guidelines
```

合并 LoRA：

```bash
PYTHONPATH=$(pwd) python src/SFT_finetuning/commons/merge_lora_weights.py 391 5 5 --with_guidelines
```

评估合并后的本地模型：

```bash
PYTHONPATH=$(pwd) python src/SFT_finetuning/evaluating/evaluate_vLLM.py LLaMA2_7B_5pos_5neg_perNE_top391NEs_TrueDef LLaMA2-chat --with_guidelines
```

Windows PowerShell 中可先设置 `PYTHONPATH`，再去掉命令前的 `PYTHONPATH=$(pwd)`：

```powershell
$env:PYTHONPATH = (Get-Location).Path
python src/SFT_finetuning/evaluating/evaluate_vLLM.py expertai/SLIMER LLaMA2-chat --with_guidelines
```

## 数据与输出

主要输入数据：

- `data/CrossNER/ner_data/`
- `data/MIT/`
- `data/eval_data_UniNER/test_data/`
- `data/pileNER/`

常见输出目录：

- `data/pileNER/{dataset_name}/`：训练脚本生成的 JSONL 数据。
- `trained_models/`：LoRA adapter。
- `merged_models/`：合并后的模型。
- `predictions/`：评估预测结果。

## 注意事项

- 训练默认使用 `meta-llama/Llama-2-7b-chat-hf`，需要 Hugging Face 访问权限。
- vLLM 推理通常需要 Linux + CUDA GPU 环境。
- 训练、合并和完整评估可能消耗较多 GPU 时间和磁盘空间。
- 原始数据建议只读，转换后的数据写入单独生成目录。

## License

本仓库使用 Apache License 2.0。详见 [LICENSE](LICENSE)。
