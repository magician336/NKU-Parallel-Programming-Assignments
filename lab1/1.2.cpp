#include <iostream>
#include <windows.h>
#include <stdlib.h>
#include <vector>
#include <cmath>

#define MAX_N 10000000

using namespace std;

double *gData = nullptr;

// 初始化数据
void init_data(int n)
{
    if (gData != nullptr)
    {
        delete[] gData;
    }
    gData = new double[n];
    for (int i = 0; i < n; i++)
    {
        gData[i] = (i % 100) * 0.01;
    }
}

void cleanGdata()
{
    if (gData != nullptr)
    {
        delete[] gData;
        gData = nullptr;
    }
}

// a) 平凡算法
double sum_naive(int n)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++)
    {
        sum += gData[i]; // 串行依赖
    }
    return sum;
}

// b) 优化算法（4x4循环展开）
double sum_optimized(int n)
{
    double sum0 = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;
    double sum3 = 0.0;

    int i = 0;
    int limit = n - (n % 4); // 计算可以整除 4 的边界

    for (; i < limit; i += 4)
    {
        sum0 += gData[i];     // 指令 A
        sum1 += gData[i + 1]; // 指令 B，不依赖 A
        sum2 += gData[i + 2]; // 指令 C，不依赖 A、B
        sum3 += gData[i + 3]; // 指令 D，不依赖 A、B、C
    }

    // 处理剩余不足 4 个的元素
    double remainder = 0.0;
    for (; i < n; i++)
    {
        remainder += gData[i];
    }

    // 级联求和
    return (sum0 + sum1) + (sum2 + sum3) + remainder;
}

// 验证结果一致性
bool verify_results(double naive_result, double opt_result)
{
    const double epsilon = 1e-7;

    // 由于浮点数精度问题，不同累加顺序可能导致微小差异
    bool valid = fabs(naive_result - opt_result) < epsilon;

    if (!valid)
    {
        cout << "警告：浮点数精度差异！" << endl;
        cout << "平凡算法结果: " << naive_result << endl;
        cout << "4x4循环展开算法结果: " << opt_result << endl;
        cout << "差异: " << fabs(naive_result - opt_result) << endl;
    }

    return valid;
}

int main()
{
    SetConsoleOutputCP(65001);

    vector<int> test_sizes = {1000, 10000, 100000, 1000000, 10000000};

    cout << "=== Lab1 实验二：n个数求和性能测试 ===" << endl;
    cout << "测试不同累加算法对超标量架构的利用程度" << endl
         << endl;

    for (int n : test_sizes)
    {
        cout << "=== 测试数据规模: " << n << " ===" << endl;

        // 初始化数据
        init_data(n);

        long long head, tail, freq;
        QueryPerformanceFrequency((LARGE_INTEGER *)&freq);

        double result_naive, result_opt;

        // 测试平凡算法
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        result_naive = sum_naive(n);
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        double time_naive = (tail - head) * 1000.0 / freq;

        // 测试4x4循环展开算法
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        result_opt = sum_optimized(n);
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        double time_opt = (tail - head) * 1000.0 / freq;

        // 验证结果
        bool correct = verify_results(result_naive, result_opt);

        // 输出结果
        cout << "平凡算法(链式累加)时间: " << time_naive << "ms" << endl;
        cout << "4x4循环展开算法时间: " << time_opt << "ms" << endl;
        cout << "加速比: " << time_naive / time_opt << "倍" << endl;
        cout << "结果正确性: " << (correct ? "通过" : "失败") << endl;
        cout << endl;

        // 清理内存
        cleanGdata();
    }

    cout << "=== 实验完成 ===" << endl;
    return 0;
}
