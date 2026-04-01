#include <iostream>
#include <windows.h>
#include <stdlib.h>

#define N 1024

using namespace std;

double col_sum[N], a[N], b[N][N];

void init(int n)
{
    SetConsoleOutputCP(65001);
    for (int i = 0; i < n; i++) {
        a[i] = i; 
        for (int j = 0; j < n; j++)
            b[i][j] = i << 2; 
    }
}

// a) 平凡算法，逐列访问元素
void col_major(int n)
{
    for (int i = 0; i < n; i++) {
        col_sum[i] = 0.0;
        for (int j = 0; j < n; j++)
            col_sum[i] += b[j][i] * a[j];
    }
}

/* b) cache优化算法
 * 在内存中，矩阵以行主序存储，由于内存中同一列的元素物理距离为n，会导致大量的缓存失效。
 * cache优化算法将逐列访问元素更改为逐行访问元素，增大了缓存命中率，提高了运行速度。
 */
void row_major(int n)
{
    for (int i = 0; i < n; i++)
        col_sum[i] = 0.0;

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            col_sum[j] += b[i][j] * a[i];
}

int main()
{
    long long head, tail, freq;

    init(N);

    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

    // 测试平凡算法
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    col_major(N);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    double time_col = (tail - head) * 1000.0 / freq;

    // 测试优化算法
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    row_major(N);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    double time_row = (tail - head) * 1000.0 / freq;

    cout << "矩阵大小：" << N << "x" << N << endl;
    cout << "平凡算法时间：" << time_col << "ms" << endl;
    cout << "cache优化算法时间：" << time_row << "ms" << endl;
    cout << "加速比：" << time_col / time_row << "倍" << endl;

    return 0;
}
