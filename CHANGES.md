# LLM 任务分解模块改进 进度和实现


## 1. 防止 LLM 幻觉

LLM 分解任务时需要知道场景里有哪些物体可以操作。目前的做法是通过 `inspect.getsource()` 把整个任务类的源码发给 LLM，让它自己从代码里理解场景结构。

问题在于，任务源码里用的是 Python 属性名（如 `self.stove`、`self.microwave`），这是开发者写代码时的命名习惯。而 Isaac Lab 场景里物体的实际 key 是另一套名字（如 `stovetop_main_group`、`microwave_main_group`），两者并不一致。

LLM 读到源码后，会把 `self.stove` 这样的属性名直接当作 `target_object` 填进 skill 序列。运行时 `env.scene["self.stove"]` 当然找不到这个物体，导致报错。

### 解决思路

最直接的解法是在 prompt 里明确告诉 LLM 这两套名字的对应关系：代码里的 `self.stove` 就是场景里的 `stovetop_main_group`，用右边这个。

为此在 `EnvExtraInfo` 里新增了 `code_name_to_scene_key` 字段，由每个 pipeline 自己维护一张映射表。这张表需要开发者手动填写——看一下对应任务类的 `_setup_scene()` 方法，找出 `self.xxx = ...` 的赋值语句，和实际 scene key 对应起来。

这是一个手动维护的缺陷，任务多了会比较麻烦。更好的长期方案是自动解析 `_setup_scene()` 里的赋值语句来建立映射，但目前手动方式已经能解决问题。

映射表通过 Jinja 模板注入 prompt，同时在 `_validate_result()` 里对 LLM 输出做后验证，如果 `target_object` 不在 `available_objects` 里则触发重试，形成双重保障。

### 修改文件

**`autosim/core/types.py`** — `EnvExtraInfo` 新增字段：
```python
code_name_to_scene_key: dict[str, str] | None = None
```
每个 pipeline 维护一张映射表，从代码属性名映射到真实 scene key，注入 LLM prompt。

**`autosim/decomposers/llm_decomposer/prompts/task_decompose.jinja`**：
- 可用物体列表改为每行一条的 bullet 格式，更清晰，LLM 不容易漏看
- 新增 `code_name_to_scene_key` 映射表段落，明确告诉 LLM 代码属性名和真实 scene key 的对应关系：
```
## CRITICAL: Code Name → Scene Key Mapping
- `self.stove` → "stovetop_main_group"
- `self.counter` → "counter_main_main_group"
```

**`autosim/decomposers/llm_decomposer/llm_decomposer.py`**：
- `_build_prompt()`：将 `code_name_to_scene_key` 传入 Jinja 模板，否则模板里的映射段落永远不会渲染
- `_validate_result()`：原来只校验 `skill_type` 是否合法，不检查 `target_object`。现在每个 skill 的 `target_object` 必须存在于 `available_objects`（在映射表里确实有这个物体），否则抛 `ValueError` 触发重试。`target_type == "position"` 的跳过（这类 skill 的 target 是方向描述如 `"up"`、`"forward"`，不是物体名）
- `decompose()`：原来没有重试，LLM 生成一次失败就直接报错。现在加了重试循环，最多尝试 `max_retries` 次，全部失败才抛异常
- `_write_debug()`：新增方法，设置了 `debug_output_dir` 后把每次 LLM 调用的 prompt 和原始 response 写成文件，这样能方便调试。重试时文件名加 `_attempt2` 后缀，不覆盖之前的记录

**`autosim/decomposers/llm_decomposer/llm_decomposer_cfg.py`**：
- 新增 `max_retries: int = 3`，控制校验失败后的最大重试次数，不硬编码在逻辑里

### 使用方式（在 pipeline 中配置）—— 要在LWBenchhub里的每一个环境都要附带一个这样的配置表

```python
def get_env_extra_info(self) -> EnvExtraInfo:
    return EnvExtraInfo(
        task_name="Robocasa-Task-KettleBoiling",
        objects=self._env.scene.keys(),
        code_name_to_scene_key={
            "self.stove": "stovetop_main_group",
            "self.counter": "counter_main_main_group",
        },
        ...
    )
```

### LW-BenchHub 各 pipeline 的映射表

| Pipeline | 映射 |
|---|---|
| kettle_boiling | `self.stove` → `stovetop_main_group`，`self.counter` → `counter_main_main_group` |
| cheesy_bread | `self.counter` → `counter_main_main_group` |
| close_oven | `self.fxtr` → `oven_main_group` |
| coffee_setup_mug | `self.coffee_machine` → `coffee_machine_main_group`，`self.counter` → `counter_main_main_group` |
| open_fridge | `self.fxtr` → `fridge_main_group` |
| pnp_counter_to_microwave | `self.microwave` → `microwave_main_group`，`self.counter` → `counter_main_main_group` |

---

## 待解决问题

### LLM 读取的任务代码过于冗余 ——— 是否需要修改目前的prompt？把整个源码发进去是不是不太合适？

当前 `_load_task_code()` 通过 `inspect.getsource()` 把整个任务类的源码（包括 reward 函数、observation 定义、物理参数等）全部发给 LLM。但任务分解实际上只需要：
- success condition（目标是什么）
- 场景物体列表（有什么可以操作）

大量无关代码增加了 prompt 长度，也增加了 LLM 误读的风险。建议后续只提取关键信息传给 LLM，而不是发送完整源码。

---

## 2. 分解结果调试输出（`debug_output_dir`）

### 问题背景

分解结果缓存在 `~/.cache/autosim/decomposer_cache/`，不在项目目录下，调试时不方便查看。LLM 的原始 prompt 和 response 之前只有 debug 级别日志，但 logger 默认 INFO 级别，实际上根本看不到。

### 修改文件 —— 希望把LLM的prompt和输出内容都输出来方便调试

**`autosim/core/decomposer.py`**：
- `DecomposerCfg` 新增字段：`debug_output_dir: str | None = None`
- 初始化时若设置了该字段则自动创建目录
- `write_cache()` 里额外将分解结果 JSON 写入 `{debug_output_dir}/{task_name}.json`
- 只在 LLM 实际调用时触发，缓存命中时不写

**`autosim/decomposers/llm_decomposer/llm_decomposer.py`**：
- `decompose()` 里每次 LLM 调用后调用 `_write_debug()`，将 prompt 和 response 写入文件

设置后每次运行在指定目录生成三个文件：
```
debug_decompose/
├── {task_name}_prompt.txt      # 发给 LLM 的完整 prompt
├── {task_name}_response.txt    # LLM 返回的原始内容
└── {task_name}.json            # 解析后的分解结果
```

### 使用方式

在 pipeline cfg 里指定目录：
```python
decomposer: LLMDecomposerCfg = LLMDecomposerCfg(debug_output_dir="./debug_decompose")
```

或运行时临时覆盖：
```python
pipeline.cfg.decomposer.debug_output_dir = "./debug_decompose"
```

---

## 3. 左右手分配（待实现）

### 问题背景

---

## 4. 加入 Grasp Metadata 和 Subtask 结构（待实现）

### 问题背景


