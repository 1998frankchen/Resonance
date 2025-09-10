# Evaluation Guide

All the evaluation scripts can be found in `scripts/eval`.

## Set-up Environment
Resonance uses [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) to evaluate most of the benchmarks.

To avoid package conflicts, please install VLMEvalKit in another conda environment:
```bash
conda create -n vlmeval python=3.10
conda activate vlmeval
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit
pip install -e .
```

To assist the evaluation, e.g. extract choice from VLM response for some benchmarks, we use [lmdeploy](https://github.com/InternLM/lmdeploy) to deploy a local judge model:
```bash
conda activate vlmeval
pip install lmdeploy
```

Then set some environment variables in `scripts/eval/config.sh`:
```bash
export conda_path="~/miniconda3/bin/activate" #path to the conda activate file
export resonance_env_name="resonance" #name of environment that installed resonance
export vlmeval_env_name="vlmeval" #name of environment that installed vlmevalkit
export judger_path="ckpts/Qwen1.5-14B-Chat" #path to the local judger checkpoint
export judger_port=23333 #service port
export judger_tp=2 #tensor parallelism size
```

You should also set some environment variables for VLMEvalKit, please refer to their [guide](https://github.com/open-compass/VLMEvalKit/blob/main/Quickstart.md#deploy-a-local-language-model-as-the-judge--choice-extractor)

## Prepare Dataset
Please put all benchmark files in `Resonance/data_dir`.
```bash
mkdir data_dir
```
### MM-Vet
```bash
cd data_dir
wget https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip
unzip mm-vet.zip
```

### SEEDBench-Image
```bash
cd data_dir
mkdir SEED-Bench
de SEED-Bench
wget https://huggingface.co/datasets/AILab-CVC/SEED-Bench/resolve/main/SEED-Bench.json?download=true -O SEED-Bench.json
wget https://huggingface.co/datasets/AILab-CVC/SEED-Bench/resolve/main/SEED-Bench-image.zip?download=true -O SEED-Bench-image.zip
unzip SEED-Bench-image.zip
```

### MMBench
```bash
cd data_dir
wget https://opencompass.openxlab.space/utils/VLMEval/MMBench_DEV_EN.tsv
```

### MathVista
```bash
cd data_dir
wget https://opencompass.openxlab.space/utils/VLMEval/MathVista_MINI.tsv
```

### MMMU
```bash
cd data_dir
wget https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv
```

### MME
```bash
cd data_dir
wget https://opencompass.openxlab.space/utils/VLMEval/MME.tsv
```

### POPE
```bash
cd data_dir
git clone https://github.com/AoiDragon/POPE.git
mkdir coco2014
cd coco2014
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip

```

## Report to MySQL
Resonance supports to report the evaluation results to MySQL database. To do this, you need to firstly create a table for each benchmark and set the columns properly.

We have already created a database [resonance.sql](./resonance.sql) for you to easily use. Just import it using:
```bash
mysql -u username -p RESONANCE< resonance.sql #the database RESONANCE must already exist.
```

Then, set some environment variables in the [config](./config.sh):
```bash
export SQL_HOST="localhost" #host name of your mysql server
export SQL_PORT=3306 # port of your mysql server
export SQL_USER="root" #username
export SQL_PASSWORD="Your Passward"
export SQL_DB="RESONANCE" # name of the database
export report_to_mysql=True #turn on reporting to MySQL
```

# Evaluate

Firstly, Set the path of checkpoint and the name of experiment as environment variables:
```bash
export CKPT=ckpts/QwenVL
export TAG="QwenVL"
```
Then run any evaluation script in the `scripts/eval`, like
```bash
bash scripts/eval/mmvet.sh
```
If you report the results to MySQL, then the value of `TAG` will be parsed into column names of the table `exps` in `resonance.sql`

For example, the table `exps` has these columns by default:`tag`,`method`,`epoch` and so on. If `TAG` is set as:
```bash
export TAG="tag:qwenvl_dpo,method:dpo,epoch=1"
```
Then the new record in `exps` will be like:
```
  tag     | method | epoch |
---------------------------
qwenvl_dpo|   dpo  |   1   |
```
The rules for writing `TAG` includes
- Use `,` as the seperator between items, without space.
- Use `:` to assign value to column whose type is string, and use `=` to assign value to column whose type is int or float.

The column `tag` is set as primary key in each table of `resonance.sql`. So you can use it to identify each experiment.

You can also evaluate multiple checkpoints on multiple benchmarks with one script:
```bash
export CKPT_PRE=ckpts/QwenVL-dpo/checkpoint-
export TAG_PRE="tag:qwenvl_dpo,step="
export SUFFIX=100,200,300
export BENCHMARKS=mmvet,mmbench,seedbench_gen
bash scripts/eval/eval_all.sh
```
Values in `SUFFIX` seperated by `,` will be joined with `CKPT_PRE` and `TAG_PRE`. The outputs will be saved in `Resonance/eval/all`.




# 评估指南

所有评估脚本都可以在 `scripts/eval` 目录中找到。

## 环境设置

Resonance使用 [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) 来评估大多数基准测试。

为避免包冲突，请在另一个conda环境中安装VLMEvalKit：

```bash
conda create -n vlmeval python=3.10
conda activate vlmeval
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit
pip install -e .
```

为了辅助评估，例如从VLM响应中提取选择（用于某些基准测试），我们使用 [lmdeploy](https://github.com/InternLM/lmdeploy) 来部署本地判断模型：

```bash
conda activate vlmeval
pip install lmdeploy
```

然后在 `scripts/eval/config.sh` 中设置一些环境变量：

```bash
export conda_path="~/miniconda3/bin/activate" #conda activate文件的路径
export resonance_env_name="resonance" #安装了resonance的环境名称
export vlmeval_env_name="vlmeval" #安装了vlmevalkit的环境名称
export judger_path="ckpts/Qwen1.5-14B-Chat" #本地判断模型检查点的路径
export judger_port=23333 #服务端口
export judger_tp=2 #张量并行大小
```

您还应该为VLMEvalKit设置一些环境变量，请参考他们的[指南](https://github.com/open-compass/VLMEvalKit/blob/main/Quickstart.md#deploy-a-local-language-model-as-the-judge--choice-extractor)

## 准备数据集

请将所有基准测试文件放在 `Resonance/data_dir` 目录中：

```bash
mkdir data_dir
```

### MM-Vet
```bash
cd data_dir
wget https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip
unzip mm-vet.zip
```

### SEEDBench-Image
```bash
cd data_dir
mkdir SEED-Bench
cd SEED-Bench
wget https://huggingface.co/datasets/AILab-CVC/SEED-Bench/resolve/main/SEED-Bench.json?download=true -O SEED-Bench.json
wget https://huggingface.co/datasets/AILab-CVC/SEED-Bench/resolve/main/SEED-Bench-image.zip?download=true -O SEED-Bench-image.zip
unzip SEED-Bench-image.zip
```

### MMBench
```bash
cd data_dir
wget https://opencompass.openxlab.space/utils/VLMEval/MMBench_DEV_EN.tsv
```

### MathVista
```bash
cd data_dir
wget https://opencompass.openxlab.space/utils/VLMEval/MathVista_MINI.tsv
```

### MMMU
```bash
cd data_dir
wget https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv
```

### MME
```bash
cd data_dir
wget https://opencompass.openxlab.space/utils/VLMEval/MME.tsv
```

### POPE
```bash
cd data_dir
git clone https://github.com/AoiDragon/POPE.git
mkdir coco2014
cd coco2014
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip
```

## 报告到MySQL

Resonance支持将评估结果报告到MySQL数据库。为此，您需要首先为每个基准测试创建表并正确设置列。

我们已经为您创建了一个数据库 [resonance.sql](./resonance.sql)，方便您使用。只需使用以下命令导入：

```bash
mysql -u username -p RESONANCE < resonance.sql #数据库RESONANCE必须已经存在
```

然后，在 [config](./config.sh) 中设置一些环境变量：

```bash
export SQL_HOST="localhost" #您的mysql服务器的主机名
export SQL_PORT=3306 #您的mysql服务器的端口
export SQL_USER="root" #用户名
export SQL_PASSWORD="Your Password" #密码
export SQL_DB="RESONANCE" #数据库名称
export report_to_mysql=True #开启报告到MySQL
```

## 评估

首先，将检查点路径和实验名称设置为环境变量：

```bash
export CKPT=ckpts/QwenVL
export TAG="QwenVL"
```

然后运行 `scripts/eval` 中的任何评估脚本，例如：

```bash
bash scripts/eval/mmvet.sh
```

如果您将结果报告到MySQL，那么 `TAG` 的值将被解析为 `resonance.sql` 中 `exps` 表的列名。

例如，`exps` 表默认有以下列：`tag`、`method`、`epoch` 等。如果 `TAG` 设置为：

```bash
export TAG="tag:qwenvl_dpo,method:dpo,epoch=1"
```

那么 `exps` 表中的新记录将如下所示：

tag | method | epoch |
qwenvl_dpo| dpo | 1 |



编写 `TAG` 的规则包括：
- 使用 `,` 作为项目之间的分隔符，不带空格
- 使用 `:` 为字符串类型的列赋值，使用 `=` 为int或float类型的列赋值

`resonance.sql` 中每个表的 `tag` 列都设置为主键。因此您可以使用它来标识每个实验。

您还可以使用一个脚本在多个基准测试上评估多个检查点：

```bash
export CKPT_PRE=ckpts/QwenVL-dpo/checkpoint-
export TAG_PRE="tag:qwenvl_dpo,step="
export SUFFIX=100,200,300
export BENCHMARKS=mmvet,mmbench,seedbench_gen
bash scripts/eval/eval_all.sh
```

`SUFFIX` 中用 `,` 分隔的值将与 `CKPT_PRE` 和 `TAG_PRE` 连接。输出将保存在 `Resonance/eval/all` 目录中。

## 支持的基准测试

### 1. MM-Vet
- **用途**: 多模态兽医评估
- **特点**: 测试模型在复杂视觉推理任务上的表现

### 2. SEEDBench-Image
- **用途**: 种子基准测试图像版本
- **特点**: 全面的视觉语言理解评估

### 3. MMBench
- **用途**: 多模态基准测试
- **特点**: 标准化的多模态模型评估

### 4. MathVista
- **用途**: 数学视觉任务
- **特点**: 测试模型在数学问题视觉化方面的能力

### 5. MMMU
- **用途**: 多模态多任务理解
- **特点**: 综合的多任务评估

### 6. MME
- **用途**: 多模态评估
- **特点**: 基础的多模态能力测试

### 7. POPE
- **用途**: 幻觉检测评估
- **特点**: 测试模型是否会产生视觉幻觉

## 评估流程

1. **环境准备**: 安装VLMEvalKit和lmdeploy
2. **数据下载**: 下载所有基准测试数据集
3. **配置设置**: 配置环境变量和数据库连接
4. **模型评估**: 运行评估脚本
5. **结果分析**: 查看和分析评估结果

## 注意事项

- 确保有足够的存储空间下载所有数据集
- 评估过程可能需要较长时间，建议使用GPU加速
- 定期备份评估结果到数据库
- 注意不同基准测试的评估指标和评分标准

