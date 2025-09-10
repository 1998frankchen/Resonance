# Customized Model
To add your own model to Resonance framework, you need to create a directory in `src/resonance/models`, and implement all the APIs we need in `__init__.py`. You can refer to other models that we have implemented in `src/resonance/models`.

## `__init__.py`
Here we list the APIs you need to implement in `__init__.py`. Most of them are inherited from our predefined abstract class, and there is no additional code needed if you have no special requirements.

- Model class: You should implement the following methods or property in the model class.
    - `default_lora_target`: A class property which is a list that contains default LoRA target modules. When `--lora_target_modules` is set to `auto` in the training command, this `default_lora_target` is used.
    - `get_vision_tower`: A class method that return the vision encoder.
    - `freeze_vision_tower`: A class method that freeze vision encoder. When `--freeze_vision_tower` is set to `True` in the training command, this method is used to freeze vision encoder.
    - `prepare_default_generation_kwargs`: A class method that return default generation kwargs dict, which is used as the generation config during evaluation.
- Processor: A subclass of `resonance.base.processor.VLProcessor`. You should implement the following abstract method:
    - `__init__`: The initialization method.
    - `tokenizer`: The class property which returns tokenizer.
    - `chat_template`: The class property which defines the chat template.
    - `image_processor`: The class property which returns the image processor.
    - `save_pretrained`: The class method which will be called after training to save the processor.
    - `process_batch_conv`: The class method which tokenizes a batch of conversations
    - `format_multimodal_prompt`: The class method which adds image placeholders to the raw prompt.
    - `remove_image_placeholder`: The class method which removes the image placeholders in given prompt.
    - `is_multimodal_prompt_valid`: The class method which checks whether the given prompt contains the image placeholder.
    - `train`: The class method which turns on training mode, e.g. setting the tokenizer to right-padding mode. It will be called before training.
    - `infer`: The class method which turns on inference mode, e.g. setting the tokenizer to left-padding mode. It will be called before evaluation.
    - `__call__`: The call method. The abstract class has implemented most of features of this method, which is able to automatically tokenize the text. What you need to do is to call it via `super().__call__` and then process images manually in your implementation.

- DataCollator: You need to implement a DataCollator for each algorithm. We have implemented abstract classes for all of them in `resonance.base.collator`. What you need to do is to create a subclass of them and process the images manually, like this:
```python
class LlavaDPODataCollatorWithPadding(VLDPODataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # first, pad everything to the same length
        padded_batch = super().__call__(features)
        imgs = [Image.open(img_path).convert("RGB") for img_path in padded_batch["img_path"]]
        padded_batch["img_input_dict"] = dict(
            pixel_values=self.processor.image_processor(images=imgs, return_tensors="pt")["pixel_values"]
        )
        return padded_batch
```
where `padded_batch["img_input_dict"]` is a dict contains all the inputs related to image (or other inputs that are not processed in `super().__call__`)

- Trainer: You need to implement a Trainer for each algorithm. We have implemented abstract classes for all of them in `resonance.base.trainer`. What you need to do is to create an empty subclass like:
```python
class LlavaSFTTRainer(VLSFTTrainer):
    ...
```
The abstract class has implemented most of features for training. If you have any special requirement, just overwrite the related class methods.

- core_mapper: Don't forget to map all the classes to the corresponding attributes of the variable `core_mapper`:
```python
core_mapper = ModelCoreMapper(
    model=LlavaForRL,
    processor=LlavaProcessor,
    dpo_collator=LlavaDPODataCollatorWithPadding,
    dpo_trainer=LlavaDPOTrainer,
    reward_model=LlavaRewardModel,
    value_model=LlavaWithValueHead,
    reward_collator=LlavaRMDataCollatorWithPadding,
    reward_trainer=LlavaRMTrainer,
    sft_collator=LlavaSFTDataCollatorWithPadding,
    sft_trainer=LlavaSFTTRainer,
    ppo_collator=LlavaPPODataCollator,
    ppo_trainer=LlavaPPOTrainer,
)
```
Resonance imports all the components via `core_mapper`.

## `auto_load.py`
You also need to add some configuration in `src/resonance/utils/auto_load.py`, so that we can map a model checkpoint to the corresponding model class.

You can find a variable `MODEL_NICKNAME_MAP` in the file. Just add an item to it, like:
```python
MODEL_NICKNAME_MAP = {
    ...
    "LlavaForConditionalGeneration": "Llava",
}
```
where the key `LlavaForConditionalGeneration` is the class name specified in the model checkpoint, and the value `LLaVA` is the **name of the directory** that contains the above `__init__.py` file.

If your model supports FlashAttention2, also add it to `FLASH_ATTN_MODELS`, like:
```python
FLASH_ATTN_MODELS = [
    ...
    "LlavaForConditionalGeneration",
]
```


# 自定义模型集成指南

要将您自己的模型添加到Resonance框架中，您需要在 `src/resonance/models` 目录下创建一个文件夹，并在 `__init__.py` 中实现我们需要的所有API。您可以参考我们在 `src/resonance/models` 中已实现的其他模型。

## `__init__.py` 实现

这里列出了您需要在 `__init__.py` 中实现的API。它们大多数都继承自我们预定义的抽象类，如果您没有特殊要求，则无需编写额外代码。

### 模型类 (Model Class)
您应该在模型类中实现以下方法或属性：

- **`default_lora_target`**: 类属性，包含默认LoRA目标模块的列表。当训练命令中 `--lora_target_modules` 设置为 `auto` 时，将使用此 `default_lora_target`。
- **`get_vision_tower`**: 类方法，返回视觉编码器。
- **`freeze_vision_tower`**: 类方法，冻结视觉编码器。当训练命令中 `--freeze_vision_tower` 设置为 `True` 时，将使用此方法冻结视觉编码器。
- **`prepare_default_generation_kwargs`**: 类方法，返回默认生成参数字典，在评估期间用作生成配置。

### 处理器 (Processor)
`resonance.base.processor.VLProcessor` 的子类。您应该实现以下抽象方法：

- **`__init__`**: 初始化方法
- **`tokenizer`**: 类属性，返回分词器
- **`chat_template`**: 类属性，定义聊天模板
- **`image_processor`**: 类属性，返回图像处理器
- **`save_pretrained`**: 类方法，训练后保存处理器时调用
- **`process_batch_conv`**: 类方法，对一批对话进行分词
- **`format_multimodal_prompt`**: 类方法，向原始提示添加图像占位符
- **`remove_image_placeholder`**: 类方法，移除给定提示中的图像占位符
- **`is_multimodal_prompt_valid`**: 类方法，检查给定提示是否包含图像占位符
- **`train`**: 类方法，开启训练模式，例如将分词器设置为右填充模式。训练前调用。
- **`infer`**: 类方法，开启推理模式，例如将分词器设置为左填充模式。评估前调用。
- **`__call__`**: 调用方法。抽象类已实现了此方法的大部分功能，能够自动对文本进行分词。您需要做的就是通过 `super().__call__` 调用它，然后在您的实现中手动处理图像。

### 数据整理器 (DataCollator)
您需要为每种算法实现一个DataCollator。我们在 `resonance.base.collator` 中为所有算法实现了抽象类。您需要做的是创建它们的子类并手动处理图像，如下所示：

```python
class LlavaDPODataCollatorWithPadding(VLDPODataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 首先，将所有内容填充到相同长度
        padded_batch = super().__call__(features)
        imgs = [Image.open(img_path).convert("RGB") for img_path in padded_batch["img_path"]]
        padded_batch["img_input_dict"] = dict(
            pixel_values=self.processor.image_processor(images=imgs, return_tensors="pt")["pixel_values"]
        )
        return padded_batch
```

其中 `padded_batch["img_input_dict"]` 是一个包含所有与图像相关的输入（或在 `super().__call__` 中未处理的其他输入）的字典。

### 训练器 (Trainer)
您需要为每种算法实现一个Trainer。我们在 `resonance.base.trainer` 中为所有算法实现了抽象类。您需要做的就是创建一个空的子类，如下所示：

```python
class LlavaSFTTrainer(VLSFTTrainer):
    ...
```

抽象类已实现了训练的大部分功能。如果您有任何特殊要求，只需重写相关的类方法。

### 核心映射器 (core_mapper)
不要忘记将所有类映射到 `core_mapper` 变量的相应属性：

```python
core_mapper = ModelCoreMapper(
    model=LlavaForRL,
    processor=LlavaProcessor,
    dpo_collator=LlavaDPODataCollatorWithPadding,
    dpo_trainer=LlavaDPOTrainer,
    reward_model=LlavaRewardModel,
    value_model=LlavaWithValueHead,
    reward_collator=LlavaRMDataCollatorWithPadding,
    reward_trainer=LlavaRMTrainer,
    sft_collator=LlavaSFTDataCollatorWithPadding,
    sft_trainer=LlavaSFTTrainer,
    ppo_collator=LlavaPPODataCollator,
    ppo_trainer=LlavaPPOTrainer,
)
```

Resonance通过 `core_mapper` 导入所有组件。

## `auto_load.py` 配置

您还需要在 `src/resonance/utils/auto_load.py` 中添加一些配置，以便我们可以将模型检查点映射到相应的模型类。

您可以在文件中找到 `MODEL_NICKNAME_MAP` 变量。只需向其添加一个项目，如下所示：

```python
MODEL_NICKNAME_MAP = {
    ...
    "LlavaForConditionalGeneration": "Llava",
}
```

其中键 `LlavaForConditionalGeneration` 是模型检查点中指定的类名，值 `Llava` 是包含上述 `__init__.py` 文件的**目录名称**。

如果您的模型支持FlashAttention2，也请将其添加到 `FLASH_ATTN_MODELS` 中，如下所示：

```python
FLASH_ATTN_MODELS = [
    ...
    "LlavaForConditionalGeneration",
]
```

## 集成步骤总结

1. **创建模型目录**: 在 `src/resonance/models/` 下创建您的模型目录
2. **实现核心组件**: 在 `__init__.py` 中实现模型、处理器、数据整理器和训练器
3. **配置映射**: 在 `auto_load.py` 中添加模型映射
4. **测试验证**: 确保所有组件正常工作

## 注意事项

- 确保所有抽象方法都被正确实现
- 图像处理需要手动实现，不能依赖基类
- 训练和推理模式需要正确切换
- 所有组件都需要在 `core_mapper` 中正确映射

