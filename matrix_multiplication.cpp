#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <string>

#define MKL
#ifdef MKL
#include "mkl.h"
#endif

using namespace std;

void generation(double* mat, size_t size)
{
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<double> uniform_distance(-2.001, 2.001);
	for (size_t i = 0; i < size * size; i++)
		mat[i] = uniform_distance(gen);
}

void matrix_mult(double* a, double* b, double* res, size_t size)
{
	int size_block_x = 4;
	int size_block_y = 16;
	int isize = size;
#pragma omp parallel for
	for (int ib = 0; ib < isize; ib += size_block_y) {
		for (int kb = 0; kb < isize; kb += size_block_x) {
			for (int jb = 0; jb < isize; jb += size_block_y) {
				int iEnd = std::min(isize, ib + size_block_y);
				int jEnd = std::min(isize, jb + size_block_y);
				int kEnd = kb + size_block_x;
				for (int i = ib; i < iEnd; i++) {
					double* iRes = res + i * isize;
					double* iB_0 = b + kb * isize;
					double* iB_1 = b + (kb + 1) * isize;
					double* iB_2 = b + (kb + 2) * isize;
					double* iB_3 = b + (kb + 3) * isize;
					double iA_0 = a[i * isize + kb];
					double iA_1 = a[i * isize + kb + 1];
					double iA_2 = a[i * isize + kb + 2];
					double iA_3 = a[i * isize + kb + 3];
#pragma omp simd
					for (int j = jb; j < jEnd; j++) {
						iRes[j] += iA_0 * iB_0[j];
						iRes[j] += iA_1 * iB_1[j];
						iRes[j] += iA_2 * iB_2[j];
						iRes[j] += iA_3 * iB_3[j];
					}
				}
			}
		}
	}
}

int main()
{
	double* mat, * mat_mkl, * a, * b, * a_mkl, * b_mkl;
	size_t size = 1000;
	chrono::time_point<chrono::system_clock> start, end;

	mat = new double[size * size];
	a = new double[size * size];
	b = new double[size * size];
	generation(a, size);
	generation(b, size);
	memset(mat, 0, size * size * sizeof(double));

#ifdef MKL     
	mat_mkl = new double[size * size];
	a_mkl = new double[size * size];
	b_mkl = new double[size * size];
	memcpy(a_mkl, a, sizeof(double) * size * size);
	memcpy(b_mkl, b, sizeof(double) * size * size);
	memset(mat_mkl, 0, size * size * sizeof(double));
#endif

	start = chrono::system_clock::now();
	matrix_mult(a, b, mat, size);
	end = chrono::system_clock::now();


	int elapsed_seconds = chrono::duration_cast<chrono::milliseconds>
		(end - start).count();
	cout << "Total time: " << elapsed_seconds / 1000.0 << " sec" << endl;

#ifdef MKL 
	start = chrono::system_clock::now();
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		size, size, size, 1.0, a_mkl, size, b_mkl, size, 0.0, mat_mkl, size);
	end = chrono::system_clock::now();

	elapsed_seconds = chrono::duration_cast<chrono::milliseconds>
		(end - start).count();
	cout << "Total time mkl: " << elapsed_seconds / 1000.0 << " sec" << endl;

	int flag = 0;
	for (unsigned int i = 0; i < size * size; i++)
		if (abs(mat[i] - mat_mkl[i]) > size * 1e-14) {
			flag = 1;
		}
	if (flag)
		cout << "fail" << endl;
	else
		cout << "correct" << endl;

	delete (a_mkl);
	delete (b_mkl);
	delete (mat_mkl);
#endif

	delete (a);
	delete (b);
	delete (mat);

	//system("pause");

	return 0;
}
