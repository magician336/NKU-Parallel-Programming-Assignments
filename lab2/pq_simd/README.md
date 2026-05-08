# PQ-SIMD 实验框架

## 实验目标

实现 **Product Quantization (PQ)** + **ADC (Asymmetric Distance Computation)** 策略，用 SIMD 加速 LUT 构建与查表累加过程，完成 ANNS 的高性能近似检索。

## 算法原理

### Product Quantization (PQ)

1. **切分**：将 d 维向量切分为 m 个子空间，每个子空间维度 dsub = d / m
2. **聚类**：对每个子空间独立进行 KMeans 聚类，得到 ksub 个类中心（默认 256）
3. **编码**：每个子空间用最近类中心的 ID（uint8）表示，最终向量压缩为 m 字节

### ADC 查询流程

**Step 1: LUT 构建**
- 对查询向量的每个子段，计算到该子空间 256 个类中心的精确距离
- 生成一张 **m × 256** 的查找表（LUT）

**Step 2: 查表累加**
- 对每个 Base 向量，根据其 m 字节编码，从 LUT 中查 m 个距离值并累加
- 得到近似距离：$\delta(x,q) = \sum_{sub=1}^{m} \text{LUT}[sub][\text{code}[sub]]$

**Step 3: 精排 (Rerank)**
- 选出查表累加后的 Top-p 候选
- 用原始 float 向量精确计算距离，取最终 Top-k

### 压缩比

原始向量：d × 4 bytes (float)
PQ 编码后：m × 1 byte (uint8)
压缩比 = 4d / m

例如 dim=96, m=4：384 bytes → 4 bytes，压缩比 **96x**

## 文件说明

| 文件 | 说明 |
|------|------|
| `simd_utils.h` | 跨平台 SIMD 封装（float + uint8） |
| `pq_quantizer.h` | PQ 量化器：KMeans 聚类、编码、LUT 构建、ADC 距离计算 |
| `search.h` | PQ 搜索算法：ADC 粗排 + Flat-SIMD 精排 |
| `main.cc` | 测试入口：训练 PQ、latency-recall trade-off、不同 m 的影响 |
| `Makefile` | 本地编译脚本 |

## 编译运行

```bash
make
./pq_simd
```

## 预期输出

1. **Latency-Recall Trade-off**：不同 p 值下的延迟、召回率、加速比
2. **子空间数影响**：不同 m 值对压缩比和召回率的影响

## 进阶优化方向

| 优化点 | 说明 | 难度 |
|--------|------|------|
| **LUT SIMD 加速** | 用 SIMD 并行计算查询子向量到多个类中心的距离 | 中 |
| **跨 Centroid 并行** | 同时处理多个类中心的距离计算 | 中 |
| **Gather 指令** | 用 SIMD gather 并行查表 | 高 |
| **FastScan** | 将 LUT 量化到寄存器内完成查表（shuffle） | 高 |
| **OPQ** | Optimized Product Quantization，旋转矩阵优化子空间划分 | 高 |
| **RaBitQ** | 理论有界误差的量化方法 | 很高 |

## 与前后阶段的关联

| 方向 | 关系 |
|------|------|
| **继承自 SQ-SIMD** | 复用 `simd_utils.h`，将逐维量化升级为子空间乘积量化 |
| **演进方向** | 结合 IVF（倒排索引）构建 IVF-PQ；或在 HNSW 上集成 PQ 压缩索引 |

## 评分参考

- 高质量完成到 LUT 构建（基础优化）：**累计 14 分**
- 高质量实现查表累加进阶优化（FastScan / gather / shuffle 等）：**累计 15 分**
