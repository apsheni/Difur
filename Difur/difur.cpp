#include <iostream>
#include <fstream>
#include <algorithm>
#include <math.h>
#include <omp.h>
#include <windows.h>

using namespace std;

// initial constants
const double E = 100.0;
const double r = 0.08;
const double sigma = 0.2;
const double eps = 0.00001;


// 5-points grid
double grid(double t, double S)
{
	auto tic = GetTickCount();
	double res, tmp, delta, dlocal, dmax = 0.0, ht = 0.05, hS = 1.0;
	int i, j, loops = 0, Nt = 10, NS = 200;
	// coefs
	double k11, k1 = 1.0 / (2 * ht*sigma*sigma*S*S);
	double k2 = 0.5;
	double k3 = r / (hS*sigma*sigma*S);
	double k4 = 2.0 * r / (sigma*sigma*S*S);
	double **u = new double *[Nt+1]();

	for (i = 0; i <= Nt; i++)
	{
		u[i] = new double[NS+1]();
	}
	for (i = 0; i <= Nt; i++)
	{
		u[i][0] = E* exp(-r * i*ht);
	}
	for (j = 1; j <= NS; j++)
	{
		u[0][j] = max(E - j*hS, 0.0);
	}

// make treads for the cycle
omp_lock_t dmax_lock;
omp_init_lock(&dmax_lock);
	do {
		dmax = 0.0;
		loops++;
#pragma omp parallel for shared(u,Nt,NS,dmax,dlocal) private(i,j,tmp,delta,k11)
		for (i = 1; i < Nt; i++)
		{
			dlocal = 0.0;
			for (j = 1; j < NS; j++)
			{
				k11 = (j < S / hS) ? k1 : 0.0;
				tmp = u[i][j];
				u[i][j] = -k11*(u[i + 1][j] - u[i - 1][j]) + k2*(u[i][j + 1] + u[i][j - 1])
							+ k3*(u[i][j + 1] - u[i][j - 1]) - k4*u[i][j];
				delta = fabs(tmp - u[i][j]);
				if (dlocal < delta) dlocal = delta;
			}
omp_set_lock(&dmax_lock);
			if (dmax < dlocal) dmax = dlocal;
omp_unset_lock(&dmax_lock);
		}
	} while (dmax > eps);
omp_destroy_lock(&dmax_lock);

	tic = GetTickCount() - tic;
	ofstream resfile("results.txt");

	res = u[(int)(t / ht - 1)][(int)(S / hS)];
	resfile << "V(" << t << "," << S << "): " << res << endl;
	resfile << "loops: " << loops << endl;
	resfile << "took " << tic << " ms" << endl;

	for (j = NS-1; j >= 0; j--)
	{
		resfile << endl;
		for (i = 0; i < Nt; i++)
		{
			resfile.width(15);
			resfile << u[i][j];
		}
	}

	resfile.close();
	for (i = 0; i < Nt; i++)
	{
		delete[] u[i];
		u[i] = NULL;
	}
	delete[] u;
	u = NULL;

	return res;
}

int main()
{
	// calculate for t = 0.25, S = 100.0
	double t = 0.25, S = 100.0;
	grid(t, S);
}