# SQ-SIMD 实验框架

## 实验目标

在 Flat-SIMD 基础上引入 **Scalar Quantization (SQ)**，实现**两阶段检索**（粗排 + 精排），在减少计算与访存开销的同时保持较高的召回率。

## 算法原理

### Scalar Quantization (SQ)

将每个维度从 **32-bit float** 量化为 **8-bit unsigned integer**：

$$\text{code}[i] = \text{round}\left( \frac{\text{vec}[i] - \min_i}{\max_i - \min_i} \times 255 \right)$$

反量化：

$$\text{vec}[i] = \text{code}[i] \times \text{scale}_i + \min_i$$

### 两阶段检索

| 阶段 | 名称 | 说明 | 数据类型 |
|------|------|------|---------|
| A | **粗排 (Coarse)** | 在量化空间计算近似距离，快速筛选 Top-p 候选 | uint8 |
| B | **精排 (Rerank)** | 对 Top-p 候选用原始 float 向量重新计算精确距离，取 Top-k | float |

### 距离策略

- **ADC**（非对称距离计算，推荐）：查询向量保持 float，Base 向量使用 uint8 编码
- **SDC**（对称距离计算）：查询向量也进行 SQ 编码，两个编码直接比较

调节参数 **p**（粗排候选数）可实现 **latency-recall trade-off**：
- p 越大 → 召回率越高，但延迟增加
- p 越小 → 延迟越低，但召回率下降

## 文件说明

| 文件 | 说明 |
|------|------|
| `simd_utils.h` | 跨平台 SIMD 封装（扩展了 `simd16u8` 用于 uint8） |
| `sq_quantizer.h` | SQ 量化器：train / encode / decode / ADC / SDC 距离计算 |
| `search.h` | SQ 搜索算法：两阶段粗排 + 精排，复用 Flat-SIMD 精排逻辑 |
| `main.cc` | 测试入口：训练量化器、不同 p 值的 latency-recall 实验 |
| `Makefile` | 本地编译脚本 |

## 编译运行

```bash
make
./sq_simd
```

## 预期输出

程序会输出不同 `p` 值下的平均延迟、Recall@10 和相对于串行的加速比，例如：

```
 p   Latency(ms)  Recall@10    Speedup
 10        0.123     0.7521       5.20
 20        0.156     0.8912       4.10
 ...
```

## 与前后阶段的关联

| 方向 | 关系 |
|------|------|
| **继承自 Flat-SIMD** | 复用 `simd_utils.h`（扩展 uint8）、精排阶段的 `l2_distance_simd` |
| **演进为 PQ-SIMD** | 将 SQ 的逐维量化升级为**子空间乘积量化**，粗排从逐维计算变为**查表累加 (LUT)** |

## 评分参考

高质量完成 SQ-SIMD 部分：**累计 7 分**
