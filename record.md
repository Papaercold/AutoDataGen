# AutoDataGen 开发记录

任务背景：X7S 双臂移动机器人执行 KettleBoiling 任务（导航 → 双臂抓水壶 → 放到灶台 → 开燃气灶）。
机器人：X7S-Joint，25 DOF（3 底盘 + 4 躯干/头 + 9 左臂 + 9 右臂）。

---

## 当前方案：EEF 朝向搜索解决 yaw 随机化问题

### 解决方案

**核心思路**：保持 EEF 目标**位置不变**，对目标**朝向**绕 Z 轴正负交替枚举候选偏移，找第一个规划成功且底盘偏转不超过 90° 的解。

**实现**（`curobo_planner.py: plan_motion`）：

```python
# 候选: [0°, +5°, -5°, +10°, -10°, ..., +175°, -175°, +180°]
candidates = [0]
for i in range(5, 181, 5):
    candidates.append(i)
    if i != 180:
        candidates.append(-i)
```

对每个候选角度：
1. 旋转主末端（link11_tip）和附加末端（link20_tip）的目标朝向
2. 调用 `plan_single`
3. `IK_FAIL` → 换下一个朝向（几何无解，继续搜索有意义）
4. `FINETUNE_TRAJOPT_FAIL` → 换下一个朝向（稍微不同的朝向可能让 trajopt 收敛）
5. 成功但 `base_yaw_joint` 偏移 >90° → 换下一个朝向
6. 成功且底盘偏移 ≤90° → 接受并返回

**底盘偏移限制的原因**：cuRobo 会把底盘大幅旋转当成解法。轨迹后处理覆写底盘关节值是错的（IK 解是联动的，强制覆写一个关节会让其他关节位置失效），因此通过筛选找一个底盘少动的解。

**缓存上次成功的偏移量**，下次 `plan_motion` 调用优先从该值开始搜索：

```python
self._last_successful_offset = angle_deg
```

### lift/push/pull 的特殊处理

`RelativeReachSkill`（lift/push/pull）的目标朝向取的是当前 EEF 的实际朝向（FK 计算）。如果 reach 用了 +80° 偏移，EEF 朝向已经旋转了 80°，再继承缓存的 80° 会叠加到 160°，导致 IK_FAIL。

**解决**：`RelativeReachSkill` 调用 `plan_motion` 前重置缓存，使搜索从 0° 重新开始：

```python
self._planner._last_successful_offset = 0
```

### IK_FAIL vs FINETUNE_TRAJOPT_FAIL

| 状态 | 含义 | 对应处理 |
|---|---|---|
| `IK_FAIL` | 逆运动学无解，找不到任何关节构型到达目标 | 换朝向重试 |
| `FINETUNE_TRAJOPT_FAIL` | IK 成功但轨迹精调不收敛，有随机性 | 换朝向重试（稍微不同的朝向可能收敛） |

---

## 其他 Bug 修复

| 问题 | 原因 | 解决 |
|---|---|---|
| `StopIteration` | `_build_iterator` 用普通迭代器，步骤数超过预设数量时耗尽 | 改用 `itertools.cycle` |
| `FINETUNE_TRAJOPT_FAIL`（固定角度时） | `reset_env` 中水壶四元数误写为 identity（0°），EEF 目标偏差 >0.4m | 改回 `[qw=0, qz=1]`（Z轴 180°） |
| `INVALID_START_STATE_JOINT_LIMITS` | lift 后关节值浮点误差轻微超出 cuRobo 限位 | `plan_motion` 入口处 `current_q.clamp(q_min, q_max)` |

---

## IK 目标姿态生成流程

```
预设姿态（物体坐标系）
     ↓ combine_frame_transforms × 物体当前位姿（含 yaw）
世界坐标系目标姿态
     ↓ subtract_frame_transforms ÷ 机器人根节点位姿
机器人根坐标系目标姿态 → cuRobo plan_single
```

---

## 修改文件汇总

| 文件 | 修改内容 |
|---|---|
| `source/autosim/autosim/core/types.py` | `_build_iterator` 改用 `itertools.cycle` |
| `source/autosim/autosim/capabilities/motion_planning/curobo/curobo_planner.py` | 关节 clamp；`plan_motion` 加朝向搜索、底盘偏移过滤、成功偏移缓存 |
| `source/autosim/autosim/skills/relative_reach.py` | 调用 `plan_motion` 前重置缓存 |
| `lw_benchhub/autosim/pipelines/kettle_boiling.py` | 水壶四元数修正；yaw 完全随机化；打印当前 yaw；移除水壶碰撞建模 |

---

## Yaw 随机化鲁棒性测试

### 阶段一：固定朝向，无朝向搜索（原始规划器）

基准朝向 Z轴 180°，偏转正 = 顺时针，负 = 逆时针。

| 偏转 | 总旋转角 | 结果 |
|---|---|---|
| -80° | 100° | ✅ |
| -70° | 110° | ✅ |
| -60° | 120° | ✅ |
| -50° | 130° | ✅ |
| -40° | 140° | ✅ |
| -30° | 150° | ✅ |
| -20° | 160° | ✅ |
| -10° | 170° | ✅ |
| 0° | 180° | ✅ |
| +10° | 190° | ✅ |
| +20° | 200° | ✅ |
| +21.5° | 201.5° | ✅ |
| +22.5° | 202.5° | ❌ |
| +25° | 205° | ❌ |
| +30° | 210° | ❌ |

### 阶段二：朝向搜索策略

| 水壶 yaw | delta | 结果 | 找到的偏移角 |
|---|---|---|---|
| 36.3° | -143.7° | ✅ | +80° |
| 100° | -80° | ✅ | 0° |
| 126.9° | -53.1° | ✅ | 0° |
