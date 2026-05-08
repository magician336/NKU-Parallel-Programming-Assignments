# Flat-SIMD 实验框架

## 实验目标

使用 SIMD 指令直接加速原始 float 向量的距离计算，建立 ANNS 搜索的串行 baseline 与 SIMD 优化版本，作为后续 SQ-SIMD、PQ-SIMD 实验的基础。

## 算法原理

对欧氏距离（或内积距离）的计算公式进行向量化：

1. 每次加载 **4 个 float** 到 128-bit SIMD 寄存器
2. 对位计算差值（L2）或乘积（InnerProduct）
3. 累加到向量寄存器（使用 FMA 如果可用）
4. 循环结束后，将 4 个累加值**水平求和**得到最终距离

### 距离公式

- **L2 Distance Squared**: $\delta(x,y) = \sum_{i=1}^{d}(x_i - y_i)^2$
- **Inner Product Distance**: $\delta(x,y) = 1 - \sum_{i=1}^{d}x_i \cdot y_i$

## 文件说明

| 文件 | 说明 |
|------|------|
| `simd_utils.h` | 跨平台 SIMD 封装（SSE / AVX / NEON），提供 `simd4f` 类型 |
| `search.h` | Flat 搜索算法（串行 + SIMD），header-only，含 Top-K 选择 |
| `main.cc` | 测试入口：生成数据、计时、对比加速比、计算 recall |
| `Makefile` | 本地编译脚本，自动检测 x86 / ARM 平台 |

## 编译运行

### Linux / macOS
```bash
make
./flat_simd
```

### Windows (MinGW)
```bash
mingw32-make
flat_simd.exe
```

### ARM 服务器（正式提交）
```bash
g++ main.cc -o main -O2 -fopenmp -lpthread -std=c++11 -march=armv8-a
bash test.sh 1 1   # SIMD实验，申请1个节点
```

## 与后续阶段的关联

| 阶段 | 复用内容 | 新增内容 |
|------|---------|---------|
| **SQ-SIMD** | `simd_utils.h`（扩展 uint8）、`search.h` 精排逻辑 | `sq_quantizer.h`（量化 + 粗排） |
| **PQ-SIMD** | `simd_utils.h`、`search.h` 精排逻辑 | `pq_quantizer.h`（PQ 编码 + ADC 查表） |

## 评分参考

高质量完成 Flat-SIMD 部分：**累计 3 分**
