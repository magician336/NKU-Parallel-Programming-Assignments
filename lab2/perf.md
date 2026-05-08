 # 1. 进入 ann 目录
  cd ~ann

  # 2. 复制 benchmark.cc 和 simd_utils.h（如果还没有）
  # 或者直接新建一个简单测试文件

  # 3. 编译（保留调试信息 -g）
  g++ -std=c++11 -O2 -g -I. benchmark.cc -o benchmark_perf

  # 4. perf stat 统计关键指标
  perf stat -e cycles,instructions,cache-misses,cache-references,L1-dcache-load-misses .benchmark_perf

  # 5. 保存结果到文件
  perf stat -e cycles,instructions,cache-misses,cache-references .benchmark_perf  perf_stat.txt 2&1

  # 6. perf record + report 查看热点
  perf record -g .benchmark_perf
  perf report --stdio  perf_report.txt 2&1
