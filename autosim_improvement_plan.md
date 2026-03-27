# AutoDataGen 任务分解优化方案

## 背景

当前 LLM 任务分解模块存在以下问题，需要针对性优化：

1. LLM 经常生成环境中不存在的物体
2. 多手臂任务缺少左右手分配，存在动作冲突
3. LLM prompt 缺少 grasp metadata 和 subtask 结构信息
4. 分解结果 JSON 不可见，调试不方便

---

## 问题 1：LLM 生成不存在的物体

### 现状分析

- `task_decompose.jinja` 已有 `CRITICAL: Available Objects in Scene` 约束，但 LLM 仍会幻觉出不存在的物体
- `_validate_result()` 只验证 `skill_type` 是否合法，没有验证 `target_object` 是否在场景中存在

### 解决方案

**方案 A：加强后验证（短期，低成本）**

在 `LLMDecomposer._validate_result()` 中增加 target_object 校验：

```python
# llm_decomposer.py _validate_result()
available_objects = set(extra_info.objects or [])
for subtask in result["subtasks"]:
    for skill in subtask["skills"]:
        if skill["target_object"] not in available_objects:
            raise ValueError(
                f"target_object '{skill['target_object']}' not in scene. "
                f"Available: {available_objects}"
            )
```

验证失败时触发重试（最多 N 次）。

**方案 B：改进 prompt（配合 A）**

在 jinja 模板中把物体列表改为编号列表，并要求 LLM 引用编号：

```
## CRITICAL: Available Objects (use EXACT names below, nothing else)
{% for obj in objects %}
- "{{ obj }}"
{% endfor %}
```

**方案 C：结构化输出（长期）**

使用支持 JSON Schema 约束的模型（GPT-4o、DeepSeek）通过 `response_format` 参数限制输出，把 `target_object` 字段定义为 `enum`，直接在模型层面限制可选值。

**推荐**：先做 A+B，解决 90% 的问题；有余力再做 C。

---

## 问题 2：左右手分配

### 现状分析

- `SkillInfo` 没有 `hand` 字段
- skill 执行时没有区分左右 EEF
- `EnvExtraInfo` 中 `ee_link_name` 只有一个，不支持双臂

### 解决方案

**Step 1：扩展数据结构**

在 `SkillInfo` 中增加 `hand` 字段：

```python
# core/types.py
@dataclass
class SkillInfo:
    step: int
    skill_type: str
    target_object: str
    target_type: str
    description: str
    hand: str = "right"  # "left" | "right" | "both"
```

在 `EnvExtraInfo` 中支持双 EEF：

```python
@dataclass
class EnvExtraInfo:
    ...
    ee_link_name: str = "right_ee_link"         # 右手（向后兼容）
    left_ee_link_name: str | None = None         # 左手（可选）
    left_robot_base_link_name: str | None = None
```

**Step 2：更新 prompt**

在 jinja 模板中加入双臂规则：

```
## Dual-Arm Assignment Rules
- Use "right" hand for primary manipulation (grasping, reaching)
- Use "left" hand for secondary tasks (holding, stabilizing, opening fixtures)
- Avoid assigning both hands to the same object simultaneously
- For pick-and-place: right hand picks object, left hand handles target fixture if needed
- Add "hand" field to each skill: "left" | "right" | "both"
```

**Step 3：skill 执行时路由到对应手臂**

在 `Pipeline._check_skill_extra_cfg()` 中根据 `skill_info.hand` 把对应的 EEF 配置注入 skill。

**需与帅英豪确认**：
- X7S 左右 EEF link 名称
- 双臂 cuRobo 规划是否需要两个独立 planner 实例还是一个联合规划

---

## 问题 3：加入 Grasp Metadata 和 Subtask 信息

### 现状分析

- `EnvExtraInfo.object_reach_target_poses` 存了抓取位姿，但没传给 LLM
- LLM 不知道每个物体有哪些合法抓取点，导致生成的 skill 序列可能不合理

### 解决方案

**Step 1：在 EnvExtraInfo 中加入可描述的 grasp metadata**

```python
@dataclass
class GraspMetadata:
    object_name: str
    num_grasp_points: int
    approach_direction: str    # "top" | "front" | "side"
    grasp_description: str     # 自然语言描述，供 LLM 参考
```

**Step 2：更新 jinja 模板，将 grasp metadata 注入 prompt**

```
{% if grasp_metadata %}
## Grasp Metadata
{% for meta in grasp_metadata %}
- **{{ meta.object_name }}**: {{ meta.num_grasp_points }} grasp point(s),
  approach from {{ meta.approach_direction }}. {{ meta.grasp_description }}
{% endfor %}
{% endif %}
```

**Step 3：Subtask 结构提示**

在 `additional_prompt_contents` 中（或 jinja 新增 section）传入任务的 subtask 骨架，让 LLM 在固定结构内填充 skill：

```
## Expected Subtask Structure
1. Pick [object] from [location]
2. Navigate to [target]
3. Place [object] on [target]
```

这样 LLM 输出的 subtask 划分更一致，便于验证。

---

## 问题 4：分解结果可见性（调试）

### 现状分析

- 缓存路径固定在 `~/.cache/autosim/decomposer_cache/`，不在项目目录下
- 没有工具直接查看/清除缓存

### 解决方案

**Step 1：支持可配置的输出目录**

在 `DecomposerCfg` 中增加 `debug_output_dir`：

```python
@configclass
class DecomposerCfg:
    cache_dir: str = "~/.cache/autosim/decomposer_cache"
    debug_output_dir: str | None = None  # 若设置，同时写到此目录
```

用户在 pipeline 中设置：

```python
cfg.decomposer.debug_output_dir = "./debug_decompose"
```

每次分解后同时写到 `./debug_decompose/{task_name}.json`，可直接在项目目录看到结果。

**Step 2：提供 CLI 工具**

```bash
# 查看某任务的缓存分解结果
python -m autosim.tools.show_cache KettleBoiling-v0

# 清除缓存，强制重新分解
python -m autosim.tools.clear_cache KettleBoiling-v0
```

**Step 3：分解结果格式化输出**

在 pipeline 日志中打印 skill 序列摘要：

```
[Decomposer] KettleBoiling-v0:
  Subtask 1 (Pick Kettle): moveto → reach → grasp → lift
  Subtask 2 (Place on Stove): moveto → reach → ungrasp
```

---

## 实施优先级

| 优先级 | 问题 | 方案 | 工作量 |
|--------|------|------|--------|
| P0 | 不存在物体 | A + B（后验证 + prompt 改进） | 小 |
| P0 | 调试可见性 | debug_output_dir + 日志摘要 | 小 |
| P1 | Grasp metadata 注入 | EnvExtraInfo 扩展 + jinja 更新 | 中 |
| P2 | 左右手分配 | SkillInfo 扩展 + prompt 规则 + 执行路由 | 大，需与帅英豪协作 |

---

## 与帅英豪的协作点

1. **左右手执行路由**：skill 执行时如何根据 `hand` 字段选择对应的 EEF link 和 planner
2. **任务分解验证框架**：如何系统性地验证 LLM 输出的 skill 序列在特定任务上的正确率
3. **双臂 cuRobo 规划方案**：两个独立 planner vs 联合规划的取舍
