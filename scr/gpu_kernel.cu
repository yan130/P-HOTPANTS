#include<stdio.h>
#include<math.h>
#include<cutil.h>
#include<cutil_math.h>
#include<cutil_inline.h>
#include<omp.h>
#include"gforg.h"
#include"defaults.h"

#define GPU_PART 0.7 //GPU working ratio in Convolving#define SIZE 1024
#define mats 53
#define WARPS 32

/* functions in alard.c or functions.c */
extern "C" double get_background(int, int, double *);
extern "C" int cutSStamp(stamp_struct *, float *);
extern "C" double make_kernel(int, int, double *);
extern "C" int sigma_clip(float *, int, double *, double *, int);
extern "C" int ludcmp_d1(double *, int, int, int *);
extern "C" void lubksb_d1(double *, int, int, int *, double *);
extern "C" int getStampStats3(float *, int, int, int, int, double *, double *,
		double *, double *, double *, double *, double *, int, int, int);

void build_matrix_first(stamp_struct *, float *, double *, double *, double *,
		int *, int *, int *, int);
void getStampSig(stamp_struct *stamp, double *vectors, double *kernelSol,
		float *imNoise, float *m1, float *m2, float *m3, int *mm1, int *mm2,
		int *mm3, int flag, int *mystamp, int * xstamp, int * ystamp);

extern __shared__ char array[];

extern "C" void startgpu() {
	int num_gpus;
	cudaGetDeviceCount(&num_gpus);
	if (num_gpus < 1) {
		printf("no CUDA capable devices were detected\n");
	}

	return;
}

extern "C" void allocateStamps(stamp_struct *stamps, double **vectors,
		double **mat, double **scprod, double **KerSol, int nStamps) {
	int i;
	mat_size = (nCompKer - 1) * nComp + nBGVectors + 1;
	/* allocating memory on GPU */
	CUDA_SAFE_CALL(
			cudaMalloc(vectors,
					sizeof(double) * (nCompKer + nBGVectors) * SIZE * nStamps));

	CUDA_SAFE_CALL(cudaMalloc(mat, sizeof(double) * mats * mats * nStamps));

	CUDA_SAFE_CALL(cudaMalloc(scprod, sizeof(double) * mats * nStamps));

	CUDA_SAFE_CALL(cudaMalloc(KerSol, sizeof(double) * (nCompTotal + 1)));
	/* Initialization */
	for (i = 0; i < nStamps; i++) {
		stamps[i].x0 = stamps[i].y0 = stamps[i].x = stamps[i].y = 0;
		stamps[i].nss = stamps[i].sscnt = 0;
		stamps[i].nx = stamps[i].ny = 0;
		stamps[i].sum = stamps[i].mean = stamps[i].median = 0;
		stamps[i].mode = stamps[i].sd = stamps[i].fwhm = 0;
		stamps[i].lfwhm = stamps[i].chi2 = 0;
		stamps[i].norm = stamps[i].diff = 0;
		stamps[i].xss = (int *) calloc(nKSStamps, sizeof(int));
		stamps[i].yss = (int *) calloc(nKSStamps, sizeof(int));
		stamps[i].krefArea = (double *) calloc(fwKSStamp * fwKSStamp,
				sizeof(double));
	}

	return;
}

extern "C" void freeStampMem(stamp_struct *stamps, double *vectors, double *mat,
		double *scprod, int nStamps) {
	/*****************************************************
	 * Free ctStamps allocation when ciStamps are used, vice versa
	 *****************************************************/
	int i;
	CUDA_SAFE_CALL(cudaFree(vectors));
	CUDA_SAFE_CALL(cudaFree(mat));
	CUDA_SAFE_CALL(cudaFree(scprod));

	if (stamps) {
		for (i = 0; i < nStamps; i++) {
			if (stamps[i].krefArea)
				free(stamps[i].krefArea);
			if (stamps[i].xss)
				free(stamps[i].xss);
			if (stamps[i].yss)
				free(stamps[i].yss);
		}
	}
}

extern "C" void cuda_init(float *tRData, float *iRData, float **dtRData,
		float **diRData) {
	ncomp1 = nCompKer - 1;
	ncomp2 = ((kerOrder + 1) * (kerOrder + 2)) / 2;
	ncomp = ncomp1 * ncomp2;
	nbg_vec = ((bgOrder + 1) * (bgOrder + 2)) / 2;

	pixStamp = fwKSStamp * fwKSStamp;
	rPixX2 = 0.5 * rPixX;
	rPixY2 = 0.5 * rPixY;
	vsize = nCompKer + nBGVectors;
	CUDA_SAFE_CALL(cudaMalloc(&d_xstamp, sizeof(int) * nStamps));
	CUDA_SAFE_CALL(cudaMalloc(&d_ystamp, sizeof(int) * nStamps));
	CUDA_SAFE_CALL(cudaMalloc(dtRData, sizeof(float) * rPixX * rPixY));
	CUDA_SAFE_CALL(cudaMalloc(diRData, sizeof(float) * rPixX * rPixY));
	CUDA_SAFE_CALL(
			cudaMemcpy(*dtRData, tRData, sizeof(float) * rPixX * rPixY,
					cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(
			cudaMemcpy(*diRData, iRData, sizeof(float) * rPixX * rPixY,
					cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMalloc(&d_wxy, sizeof(double) * nStamps * ncomp2));
	CUDA_SAFE_CALL(
			cudaMalloc(&d_matrix,
					sizeof(double) * (mat_size + 1) * (mat_size + 1)));
	CUDA_SAFE_CALL(cudaMalloc(&d_ksol, sizeof(double) * (nCompTotal + 1)));
	CUDA_SAFE_CALL(cudaMalloc(&d_sflag, sizeof(int) * nStamps));
	CUDA_SAFE_CALL(cudaMalloc(&d_fx, sizeof(double) * nStamps));
	CUDA_SAFE_CALL(cudaMalloc(&d_fy, sizeof(double) * nStamps));
	CUDA_SAFE_CALL(cudaMalloc(&dkernel_vec, sizeof(double) * nCompKer * SIZE));

	CUDA_SAFE_CALL(cudaMallocHost(&fx, sizeof(double) * nCompKer * fwKernel));
	CUDA_SAFE_CALL(cudaMallocHost(&fy, sizeof(double) * nCompKer * fwKernel));
	CUDA_SAFE_CALL(cudaMallocHost(&mystamp, sizeof(int) * nStamps));
	CUDA_SAFE_CALL(cudaMallocHost(&xstamp, sizeof(int) * nStamps));
	CUDA_SAFE_CALL(cudaMallocHost(&ystamp, sizeof(int) * nStamps));
	CUDA_SAFE_CALL(cudaMallocHost(&indx, sizeof(int) * nStamps * mats));

}

void filter(double * filter_x, double * filter_y, int *ren, int n, int deg_x,
		int deg_y, int ig) {
	/*****************************************************
	 * Creates kernel sized entry for kernel_vec for each kernel degree 
	 *   Mask of filter_x * filter_y, filter = exp(-x**2 sig) * x^deg 
	 *   Subtract off kernel_vec[0] if n > 0
	 * NOTE: this does not use any image
	 ******************************************************/

	int k, dx, dy, ix;
	double sum_x, sum_y, x, qe;
	ren[n] = 0;
	dx = (deg_x / 2) * 2 - deg_x;
	dy = (deg_y / 2) * 2 - deg_y;
	sum_x = sum_y = 0.0;

	for (ix = 0; ix < fwKernel; ix++) {
		x = (double) (ix - hwKernel);
		k = ix + n * fwKernel;
		qe = exp(-x * x * sigma_gauss[ig]);
		filter_x[k] = qe * pow(x, deg_x);
		filter_y[k] = qe * pow(x, deg_y);
		sum_x += filter_x[k];
		sum_y += filter_y[k];
	}

	sum_x = 1. / sum_x;
	sum_y = 1. / sum_y;

	if (dx == 0 && dy == 0) {
		for (ix = 0; ix < fwKernel; ix++) {
			filter_x[ix + n * fwKernel] *= sum_x;
			filter_y[ix + n * fwKernel] *= sum_y;
		}
		if (n > 0)
			ren[n] = 1;
	}
}

__global__
void kernel_vector(double *filter_x, double *filter_y, double *kernel_vec,
		int fwKernel) {
	kernel_vec[threadIdx.x + fwKernel * threadIdx.y + blockIdx.x * SIZE] =
			filter_x[threadIdx.x + blockIdx.x * fwKernel]
					* filter_y[threadIdx.y + blockIdx.x * fwKernel];
}

__global__
void kernel_vector_fix(int *ren, double *kernel_vec, int fwKernel) {
	if (ren[blockIdx.x] == 1)
		kernel_vec[threadIdx.x + blockIdx.x * SIZE] -= kernel_vec[threadIdx.x];
}

extern "C" void getKernelVec() {
	/*****************************************************
	 * Fills kernel_vec with kernel weight filter, called only once
	 *****************************************************/

	int *ig = (int*) malloc(sizeof(int) * nCompKer);
	int *idegx = (int*) malloc(sizeof(int) * nCompKer);
	int *idegy = (int*) malloc(sizeof(int) * nCompKer);
	int *ren = (int*) malloc(sizeof(int) * nCompKer);
	int n = 0;
	int i, j, k;
	double *filter_x, *filter_y;

	CUDA_SAFE_CALL(
			cudaMallocHost(&filter_x, sizeof(double) * nCompKer * fwKernel));
	CUDA_SAFE_CALL(
			cudaMallocHost(&filter_y, sizeof(double) * nCompKer * fwKernel));

	for (i = 0; i < ngauss; i++) {
		for (j = 0; j <= deg_fixe[i]; j++) {
			for (k = 0; k <= deg_fixe[i] - j; k++) {

				filter(filter_x, filter_y, ren, n, j, k, i);
				ig[n] = i;
				idegx[n] = j;
				idegy[n] = k;
				n++;
			}
		}
	}

	CUDA_SAFE_CALL(cudaMalloc(&dig, sizeof(int) * nCompKer));
	CUDA_SAFE_CALL(cudaMalloc(&didegx, sizeof(int) * nCompKer));
	CUDA_SAFE_CALL(cudaMalloc(&didegy, sizeof(int) * nCompKer));
	CUDA_SAFE_CALL(cudaMalloc(&dren, sizeof(int) * nCompKer));
	CUDA_SAFE_CALL(
			cudaMalloc(&dfilter_x, sizeof(double) * nCompKer * fwKernel));
	CUDA_SAFE_CALL(
			cudaMalloc(&dfilter_y, sizeof(double) * nCompKer * fwKernel));

	CUDA_SAFE_CALL(
			cudaMemcpy(dig, ig, sizeof(int) * nCompKer,
					cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(
			cudaMemcpy(didegx, idegx, sizeof(int) * nCompKer,
					cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(
			cudaMemcpy(didegy, idegy, sizeof(int) * nCompKer,
					cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(
			cudaMemcpy(dren, ren, sizeof(int) * nCompKer,
					cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(
			cudaMemcpy(dfilter_x, filter_x,
					sizeof(double) * nCompKer * fwKernel,
					cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(
			cudaMemcpy(dfilter_y, filter_y,
					sizeof(double) * nCompKer * fwKernel,
					cudaMemcpyHostToDevice));

	dim3 threads(fwKernel, fwKernel);

	kernel_vector<<<nCompKer, threads>>>(dfilter_x, dfilter_y, dkernel_vec, fwKernel);

	kernel_vector_fix<<<nCompKer,fwKernel*fwKernel>>>(dren, dkernel_vec, fwKernel);

	cudaFree (dig);
	cudaFree (didegx);
	cudaFree (didegy);
	cudaFree (dren);
	cudaFree (dfilter_x);
	cudaFree (dfilter_y);
	cudaFreeHost(filter_x);
	cudaFreeHost(filter_y);
	free(ig);
	free(idegx);
	free(idegy);
	free(ren);
	return;
}

__global__ void gxy_conv_stamp1(float *dtemp, double *dfilter_y, float *dimage,
		int *d_xstamp, int *d_ystamp, int hwKSStamp, int hwKernel,
		int fwKSStamp, int fwKernel, int rPixX) {

	float imc;
	int n = blockIdx.x * 0.25 + blockIdx.y * gridDim.x * 0.25;
	int xi = d_xstamp[blockIdx.y];
	int yi = d_ystamp[blockIdx.y];
	int yc;
	imc = 0.0;
	__shared__
	double filter_y[D_HWKERNEL * 2 + 1];
	__shared__
	float image[(D_HWKERNEL * 2 + 1 + D_HWKSSTAMP * 2 + 1) / 4][(D_HWKSSTAMP * 2
			+ 1) * 2];
	if (threadIdx.x < (2 * hwKernel + 1) && threadIdx.y == 0)
		filter_y[threadIdx.x] = dfilter_y[threadIdx.x
				+ (int) (blockIdx.x * 0.25) * fwKernel];
	__syncthreads();

	if (blockIdx.x % 4 == 0) {
		image[threadIdx.y][threadIdx.x] = dimage[threadIdx.y + xi - hwKSStamp
				- hwKernel + rPixX * (threadIdx.x + yi - hwKSStamp - hwKernel)];

		if ((threadIdx.y + xi - hwKSStamp - hwKernel) < rPixX
				&& (threadIdx.x + yi + blockDim.y - hwKSStamp - hwKernel)
						< rPixX)
			image[threadIdx.y][threadIdx.x + blockDim.x] = dimage[threadIdx.y
					+ xi - hwKSStamp - hwKernel
					+ rPixX
							* (threadIdx.x + yi - hwKSStamp - hwKernel
									+ blockDim.x)];
		__syncthreads();

		for (yc = -hwKernel; yc <= hwKernel; yc++) {
			imc += image[threadIdx.y][threadIdx.x + hwKernel + yc]
					* filter_y[hwKernel - yc];
		}
		dtemp[threadIdx.y + threadIdx.x * (fwKSStamp + fwKernel) + n * 2 * SIZE] =
				imc;
	}

	if (blockIdx.x % 4 == 1) {
		image[threadIdx.y][threadIdx.x] = dimage[threadIdx.y + blockDim.y + xi
				- hwKSStamp - hwKernel
				+ rPixX * (threadIdx.x + yi - hwKSStamp - hwKernel)];

		if ((threadIdx.y + blockDim.y + xi - hwKSStamp - hwKernel) < rPixX
				&& (threadIdx.x + yi + blockDim.y - hwKSStamp - hwKernel)
						< rPixX)
			image[threadIdx.y][threadIdx.x + blockDim.x] = dimage[threadIdx.y
					+ blockDim.y + xi - hwKSStamp - hwKernel
					+ rPixX
							* (threadIdx.x + yi - hwKSStamp - hwKernel
									+ blockDim.x)];
		__syncthreads();

		for (yc = -hwKernel; yc <= hwKernel; yc++) {
			imc += image[threadIdx.y][threadIdx.x + hwKernel + yc]
					* filter_y[hwKernel - yc];
		}

		dtemp[threadIdx.y + blockDim.y + threadIdx.x * (fwKSStamp + fwKernel)
				+ n * 2 * SIZE] = imc;
	}

	if (blockIdx.x % 4 == 2) {
		image[threadIdx.y][threadIdx.x] = dimage[threadIdx.y + blockDim.y * 2
				+ xi - hwKSStamp - hwKernel
				+ rPixX * (threadIdx.x + yi - hwKSStamp - hwKernel)];
		if ((threadIdx.y + blockDim.y * 2 + xi - hwKSStamp - hwKernel) < rPixX
				&& (threadIdx.x + yi + blockDim.y - hwKSStamp - hwKernel)
						< rPixX)
			image[threadIdx.y][threadIdx.x + blockDim.x] = dimage[threadIdx.y
					+ blockDim.y * 2 + xi - hwKSStamp - hwKernel
					+ rPixX
							* (threadIdx.x + yi - hwKSStamp - hwKernel
									+ blockDim.x)];
		__syncthreads();

		for (yc = -hwKernel; yc <= hwKernel; yc++) {
			imc += image[threadIdx.y][threadIdx.x + hwKernel + yc]
					* filter_y[hwKernel - yc];
		}

		dtemp[threadIdx.y + blockDim.y * 2
				+ threadIdx.x * (fwKSStamp + fwKernel) + n * 2 * SIZE] = imc;
	}

	if (blockIdx.x % 4 == 3) {
		image[threadIdx.y][threadIdx.x] = dimage[threadIdx.y + blockDim.y * 3
				+ xi - hwKSStamp - hwKernel
				+ rPixX * (threadIdx.x + yi - hwKSStamp - hwKernel)];
		if ((threadIdx.y + blockDim.y * 3 + xi - hwKSStamp - hwKernel) < rPixX
				&& (threadIdx.x + yi + blockDim.y - hwKSStamp - hwKernel)
						< rPixX)
			image[threadIdx.y][threadIdx.x + blockDim.x] = dimage[threadIdx.y
					+ blockDim.y * 3 + xi - hwKSStamp - hwKernel
					+ rPixX
							* (threadIdx.x + yi - hwKSStamp - hwKernel
									+ blockDim.x)];
		__syncthreads();
		for (yc = -hwKernel; yc <= hwKernel; yc++) {
			imc += image[threadIdx.y][threadIdx.x + hwKernel + yc]
					* filter_y[hwKernel - yc];
		}

		dtemp[threadIdx.y + blockDim.y * 3
				+ threadIdx.x * (fwKSStamp + fwKernel) + n * 2 * SIZE] = imc;
	}
}

__global__ void gxy_conv_stamp2(double *vectors, float *dtemp, double *filter_x,
		int hwKernel, int fwKSStamp, int fwKernel, int versize) {

	double imc;
	const int n = blockIdx.x + blockIdx.y * versize;

	int xc;
	imc = 0.0;

	for (xc = -hwKernel; xc <= hwKernel; xc++) {

		imc += (double) dtemp[threadIdx.x + xc + hwKernel
				+ threadIdx.y * (fwKSStamp + fwKernel)
				+ 2 * SIZE * (blockIdx.x + blockIdx.y * gridDim.x)]
				* filter_x[hwKernel - xc + (blockIdx.x) * fwKernel];
	}

	vectors[threadIdx.x + threadIdx.y * fwKSStamp + n * SIZE] = imc;

}

__global__ void vector_fix(int *ren, double *vectors, int fwKSStamp,
		int versize) {
	int m = blockIdx.x * 0.25;
	int l = (blockIdx.x % 4) * SIZE / 4;
	double q;
	if (ren[m] == 1) {
		q = vectors[threadIdx.x + l + (m + blockIdx.y * versize) * SIZE]
				- vectors[threadIdx.x + l + blockIdx.y * versize * SIZE];
		vectors[threadIdx.x + l + (m + blockIdx.y * versize) * SIZE] = q;
	}
}

__global__ void fill_vec(double *vectors, int *xstamp, int *ystamp,
		float rPixX2, float rPixY2, int nCompKer, int fwKSStamp, int hwKSStamp,
		int vsize, int bgOrder) {
	double xf = (float) (xstamp[blockIdx.y] + threadIdx.x - hwKSStamp - rPixX2)
			/ rPixX2;
	double yf = (float) (ystamp[blockIdx.y] + blockIdx.x - hwKSStamp - rPixY2)
			/ rPixY2;
	int i, j;
	int nv = nCompKer;
	double ax, ay;
	ax = 1.0;
	for (i = 0; i <= bgOrder; i++) {
		ay = 1.0;
		for (j = 0; j <= bgOrder - i; j++) {
			vectors[threadIdx.x + blockIdx.x * fwKSStamp + nv * SIZE
					+ blockIdx.y * vsize * SIZE] = ax * ay;
			ay *= yf;
			++nv;
		}
		ax *= xf;
	}
}

__global__ void build_matrix0(double *mat, double *vectors, int vsize, int half,
		int pixStamp, int msize) {
	/*****************************************************
	 * Build least squares matrix for each stamp
	 *****************************************************/
	const int n = blockIdx.x / 3;
	int x = threadIdx.x % half;
	int y = threadIdx.x / half;
	int vtotal = y * vsize + x;
	int ntotal = x * SIZE + y + n * SIZE * vsize;
	int k, m;
	double q = 0.0;

	double *vec = (double*) array;

	if (blockIdx.x % 3 == 0) {
		for (m = 0; m <= pixStamp; m += half) {
			vec[vtotal] = vectors[ntotal + m];
			__syncthreads();

			for (k = 0; k < half; k++)
				if (m + k < pixStamp)
					q += vec[k * vsize + x] * vec[k * vsize + y];

			__syncthreads();
		}
		mat[x + 1 + msize * (y + 1) + n * msize * msize] = q;

	}

	if (blockIdx.x % 3 == 1) {
		for (m = 0; m <= pixStamp; m += half) {
			vec[vtotal] = vectors[ntotal + m];
			vec[vtotal + half] = vectors[ntotal + m + half * SIZE];
			__syncthreads();

			for (k = 0; k < half; k++)
				if (m + k < pixStamp)
					q += vec[k * vsize + x + half] * vec[k * vsize + y];
			__syncthreads();
		}
		mat[x + 1 + half + msize * (y + 1) + n * msize * msize] = q;
		mat[y + 1 + msize * (x + 1 + half) + n * msize * msize] = q;
	}

	if (blockIdx.x % 3 == 2) {
		for (m = 0; m <= pixStamp; m += half) {
			vec[vtotal] = vectors[ntotal + m + half * SIZE];

			__syncthreads();

			for (k = 0; k < half; k++)
				if (m + k < pixStamp)
					q += vec[k * vsize + x] * vec[k * vsize + y];
			__syncthreads();
		}
		mat[x + 1 + half + msize * (y + 1 + half) + n * msize * msize] = q;
	}
}

__global__ void build_scprod0(double *scprod, double *vec, float *image,
		int *xstamp, int *ystamp, int fwKSStamp, int hwKSStamp, int rPixX,
		int nvec, int nC) {
	/*****************************************************
	 * Build the right side of each stamp's least squares matrix
	 *    stamp.scprod = degree of kernel fit + 1 bg term
	 *****************************************************/
	const int n = blockIdx.x;
	const int i = threadIdx.x;
	const int xi = xstamp[n];
	const int yi = ystamp[n];
	double p0 = 0.0;
	int xc, yc, k;
	/* Do eqn 4. in Alard */
	for (xc = -hwKSStamp; xc <= hwKSStamp; xc++) {
		for (yc = -hwKSStamp; yc <= hwKSStamp; yc++) {
			k = xc + hwKSStamp + fwKSStamp * (yc + hwKSStamp);
			p0 += vec[i * SIZE + k + n * SIZE * nvec]
					* (double) image[xc + xi + rPixX * (yc + yi)];
		}
	}

	scprod[i + 1 + n * mats] = p0;

}

extern "C" int fillStamp(stamp_struct *stamp, double *vectors, double *mat,
		double *scprod, float *imConv, float *imRef, float *image, int *xstamp,
		int *ystamp, int nStamps) {
	/*****************************************************
	 * Fills stamp->vectors with convolved images, and 
	 *   pixel indices multiplied by each other for background fit 
	 *****************************************************/

	float * dtemp;

	int k;
	for (k = 0; k < nStamps; k++) {
		if (verbose >= 1)
			fprintf(stderr,
					"    xs  : %4i ys  : %4i sig: %6.3f sscnt: %4i nss: %4i \n",
					stamp[k].x, stamp[k].y, stamp[k].chi2, stamp[k].sscnt,
					stamp[k].nss);
		if (stamp[k].sscnt >= stamp[k].nss) {
			/* have gone through all the good substamps, reject this stamp */
			/*if (verbose >= 2) fprintf(stderr, "    ******** REJECT stamp (out of substamps)\n");*/
			if (verbose >= 1)
				fprintf(stderr, "        Reject stamp\n");
			return 1;
		}
		if (cutSStamp(&stamp[k], image))
			return 1;
	}

	/* stores kernel weight mask for each order */
	CUDA_SAFE_CALL(
			cudaMalloc(&dtemp, sizeof(float) * nStamps * nCompKer * SIZE * 2));
	CUDA_SAFE_CALL(
			cudaMemset(dtemp, 0,
					sizeof(float) * nStamps * nCompKer * SIZE * 2));
	CUDA_SAFE_CALL(
			cudaMemset(vectors, 0, sizeof(double) * nStamps * vsize * SIZE));
	CUDA_SAFE_CALL(cudaMemset(mat, 0, sizeof(double) * nStamps * mats * mats));
	CUDA_SAFE_CALL(cudaMemset(scprod, 0, sizeof(double) * nStamps * mats));
	CUDA_SAFE_CALL(
			cudaMemcpy(d_xstamp, xstamp, sizeof(int) * nStamps,
					cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(
			cudaMemcpy(d_ystamp, ystamp, sizeof(int) * nStamps,
					cudaMemcpyHostToDevice));

	dim3 thread2(fwKSStamp, fwKSStamp);
	dim3 thread1(fwKSStamp, (fwKSStamp + fwKernel) * 0.25);
	dim3 block1(nCompKer * 4, nStamps);
	dim3 blocks(nCompKer, nStamps);
	dim3 thread3(vsize / 2, vsize / 2);
	dim3 block3(fwKSStamp, nStamps);

	gxy_conv_stamp1<<<block1, thread1,(fwKernel+62*13*0.5)*sizeof(double)>>>( dtemp, dfilter_y, imConv, d_xstamp, d_ystamp,hwKSStamp, hwKernel, fwKSStamp, fwKernel,rPixX);

	gxy_conv_stamp2<<<blocks,thread2>>>(vectors, dtemp, dfilter_x, hwKernel, fwKSStamp, fwKernel ,vsize);

	CUDA_SAFE_CALL(cudaFree(dtemp));

	vector_fix<<<block1 , SIZE * 0.25>>>( dren ,vectors, fwKSStamp,vsize);

	/* get the krefArea data */
	/* fill stamp->vectors[nvec+++] with x^(bg) * y^(bg) for background fit*/
	fill_vec<<< block3, fwKSStamp >>>(vectors, d_xstamp, d_ystamp, rPixX2, rPixY2, nCompKer, fwKSStamp, hwKSStamp, vsize, bgOrder);

	CUDA_SAFE_CALL (cudaThreadSynchronize());

	/* build stamp->mat from stamp->vectors*/
    build_matrix0<<<nStamps*3, vsize*vsize*0.25, sizeof(double)*vsize*vsize/2>>>( mat, vectors , vsize , vsize/2 , pixStamp, mats );

	/* build stamp->scprod from stamp->vectors and imRef  */

	build_scprod0<<<nStamps,vsize>>>(scprod, vectors, imRef, d_xstamp,d_ystamp, fwKSStamp, hwKSStamp, rPixX, vsize ,nC);
	cudaFree(dtemp);

	return 0;
}

extern "C" double check_stamps(stamp_struct *stamps, double *vectors,
		double *mat, double *scprod, int nStamps, float *imRef,
		float *imNoise) {
	/*****************************************************
	 * Fit each stamp independently, reject significant outliers
	 *    Next fit good stamps globally
	 *    Returns a merit statistic, smaller for better fits
	 *****************************************************/

	int i, mcnt1, mcnt2, mcnt3;
	double sum = 0, kmean, kstdev;
	double merit1, merit2, merit3, sig1, sig2, sig3;
	float * m1, *m2, *m3, *ks;

	int ntestStamps;

	/* kernel sum */
	ks = (float *) calloc(nStamps, sizeof(float));

	double *temp_matrix, *testKerSol;
	CUDA_SAFE_CALL(
			cudaMallocHost(&temp_matrix,
					sizeof(double) * nStamps * mats * mats));
	CUDA_SAFE_CALL(
			cudaMemcpy(temp_matrix, mat, sizeof(double) * nStamps * mats * mats,
					cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(
			cudaMallocHost(&testKerSol, sizeof(double) * nStamps * mats));
	CUDA_SAFE_CALL(
			cudaMemcpy(testKerSol, scprod, sizeof(double) * nStamps * mats,
					cudaMemcpyDeviceToHost));

	if (verbose >= 2)
		fprintf(stderr, " **Mat_size0: %i ncomp2: %i ncomp1: %i nbg_vec: %i \n",
				mat_size, ncomp2, ncomp1, nbg_vec);
#pragma omp parallel for private( sum)  
	for (i = 0; i < nStamps; i++) {

		/* extract check_mat to solve one particular stamp */

		/* fit stamp, the constant kernel coefficients end up in check_vec */
		ludcmp_d1(temp_matrix + i * mats * mats, mats - 3, mats,
				indx + i * mats);
		lubksb_d1(temp_matrix + i * mats * mats, mats - 3, mats,
				indx + i * mats, testKerSol + i * mats);

		/* find kernel sum */
		sum = testKerSol[1 + i * mats];
		check_stack[i] = sum;
		stamps[i].norm = sum;
		ks[i] = sum;

		if (verbose >= 2)
			fprintf(stderr, "    # %d    xss: %4i yss: %4i  ksum: %f\n", i,
					stamps[i].xss[stamps[i].sscnt],
					stamps[i].yss[stamps[i].sscnt], sum);
	}

	sigma_clip(ks, nStamps, &kmean, &kstdev, 10);

	fprintf(stderr,
			"    %.1f sigma clipped mean ksum : %.3f, stdev : %.3f, n : %i\n",
			kerSigReject, kmean, kstdev, nStamps);
	/* so we need some way to reject bad stamps here in the first test,
	 we decided to use kernel sum.  is there a better way?  part of
	 the trick is that if some things are variable, you get different
	 kernel sums, but the subtraction itself should come out ok. */
	/* stamps.diff : delta ksum in sigma */

	/* here we want to reject high sigma points on the HIGH and LOW
	 side, since we want things with the same normalization */
	for (i = 0; i < nStamps; i++) {
		stamps[i].diff = fabs((stamps[i].norm - kmean) / kstdev);
	}

	/*****************************************************
	 * Global fit for kernel solution
	 *****************************************************/

	/* do only if necessary */
	if ((strncmp(forceConvolve, "b", 1) == 0)) {

		/* allocate fitting matrix  */
		/* first find out how many good stamps to allocate */
		ntestStamps = 0;

		for (i = 0; i < nStamps; i++)
			if (stamps[i].diff < kerSigReject) {
				mystamp[ntestStamps] = i;
				xstamp[ntestStamps] = stamps[i].xss[0];
				ystamp[ntestStamps] = stamps[i].yss[0];
				ntestStamps++;

			} else {
				if (verbose >= 2)
					fprintf(stderr,
							"    # %d    skipping xss: %4i yss: %4i ksum: %f sigma: %f\n",
							i, stamps[i].xss[stamps[i].sscnt],
							stamps[i].yss[stamps[i].sscnt], stamps[i].norm,
							stamps[i].diff);
			}

		CUDA_SAFE_CALL(
				cudaMemcpy(d_sflag, mystamp, sizeof(int) * ntestStamps,
						cudaMemcpyHostToDevice));

		/* finally do fit */
		if (verbose >= 2)
			fprintf(stderr, " Expanding Test Matrix For Fit\n");

		build_matrix_first(stamps, imRef, vectors, mat, scprod, d_sflag, xstamp,
				ystamp, ntestStamps);

		CUDA_SAFE_CALL(
				cudaMemcpy(temp_matrix, d_matrix,
						(mat_size + 1) * sizeof(double) * (mat_size + 1),
						cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(
				cudaMemcpy(testKerSol, d_ksol,
						sizeof(double) * (nCompTotal + 1),
						cudaMemcpyDeviceToHost));

		ludcmp_d1(temp_matrix, mat_size, mat_size + 1, indx);
		lubksb_d1(temp_matrix, mat_size, mat_size + 1, indx, testKerSol);

		cudaFreeHost(temp_matrix);

		double *temp_kvec = (double*) malloc(
				(sizeof(double) * nCompKer * SIZE));
		CUDA_SAFE_CALL(
				cudaMemcpy(temp_kvec, dkernel_vec,
						(sizeof(double) * nCompKer * SIZE),
						cudaMemcpyDeviceToHost));
		for (i = 0; i < nCompKer; i++) {
			kernel_vec[i] = (double *) malloc(
					fwKernel * fwKernel * sizeof(double));

			for (int j = 0; j < fwKernel * fwKernel; j++) {
				kernel_vec[i][j] = temp_kvec[i * SIZE + j];
			}

		}
		free(temp_kvec);
		/* get the kernel sum to normalize figures of merit! */
		kmean = make_kernel(0, 0, testKerSol);
		/* determine figure of merit from good stamps */

		/* average of sum (diff**2 / value), ~variance */
		m1 = (float *) calloc(ntestStamps, sizeof(float));

		/* standard deviation of pixel distribution */
		m2 = (float *) calloc(ntestStamps, sizeof(float));

		/* noise sd based on histogram distribution width */
		m3 = (float *) calloc(ntestStamps, sizeof(float));

		getStampSig(stamps, vectors, testKerSol, imNoise, m1, m2, m3, &mcnt1,
				&mcnt2, &mcnt3, ntestStamps, mystamp, xstamp, ystamp);

		sigma_clip(m1, mcnt1, &merit1, &sig1, 10);
		sigma_clip(m2, mcnt2, &merit2, &sig2, 10);
		sigma_clip(m3, mcnt3, &merit3, &sig3, 10);
//printf("%f--%f\n", kmean, merit1);
		/* normalize by kernel sum */
		merit1 /= kmean;
		merit2 /= kmean;
		merit3 /= kmean;

		/* clean up this mess */
		cudaFreeHost(testKerSol);
		free(m1);
		free(m2);
		free(m3);
		free(ks);
		/* average value of figures of merit across stamps */
		fprintf(stderr,
				"    <var_merit> = %.3f, <sd_merit> = %.3f, <hist_merit> = %.3f\n",
				merit1, merit2, merit3);
		/* return what is asked for if possible, if not use backup */
		if (strncmp(figMerit, "v", 1) == 0) {
			if (mcnt1 > 0) {
				return merit1;
			} else if (mcnt2 > 0) {
				return merit2;
			} else if (mcnt3 > 0) {
				return merit3;
			} else {
				return 666;
			}
		} else if (strncmp(figMerit, "s", 1) == 0) {
			if (mcnt2 > 0) {
				return merit2;
			} else if (mcnt1 > 0) {
				return merit1;
			} else if (mcnt3 > 0) {
				return merit3;
			} else {
				return 666;
			}
		} else if (strncmp(figMerit, "h", 1) == 0) {
			if (mcnt3 > 0) {
				return merit3;
			} else if (mcnt1 > 0) {
				return merit1;
			} else if (mcnt2 > 0) {
				return merit2;
			} else {
				return 666;
			}
		}
	} else
		return 0;

	return 0;
}

__global__ void gmake_model(double *vector, double *d_xf, double *d_yf,
		double *kernelSol, float *csModel, int * goodstamp, int nCompKer,
		int kerOrder, int mat_size, int vsize) {
	/*****************************************************
	 * Create a model of the convolved image
	 *****************************************************/

	short int i1, k, ix, iy;
	double ax, ay, coeff;
	double *kernelsol = (double*) array;
	short int n = goodstamp[blockIdx.x];
	if (threadIdx.x <= mat_size)
		kernelsol[threadIdx.x] = kernelSol[threadIdx.x];
	double xf, yf;
	xf = d_xf[blockIdx.x];
	yf = d_yf[blockIdx.x];
	coeff = kernelsol[1];
	csModel[threadIdx.x + blockIdx.x * SIZE] += coeff
			* vector[threadIdx.x + n * vsize];

	k = 2;
	for (i1 = 1; i1 < nCompKer; i1++) {
		// vector = stamp->vectors[i1];
		coeff = 0.0;
		ax = 1.0;
		for (ix = 0; ix <= kerOrder; ix++) {
			ay = 1.0;
			for (iy = 0; iy <= kerOrder - ix; iy++) {
				coeff += kernelsol[k++] * ax * ay;
				ay *= yf;
			}
			ax *= xf;
		}
		csModel[threadIdx.x + blockIdx.x * SIZE] += coeff
				* vector[threadIdx.x + i1 * SIZE + n * vsize];
	}
}

void getStampSig(stamp_struct *stamp, double * vectors, double *kernelSol,
		float *imNoise, float *m1, float *m2, float *m3, int *mm1, int *mm2,
		int *mm3, int flag, int *mystamp, int * xstamp, int * ystamp) {
	int i, j, idx, nsig, xRegion, yRegion, xRegion2, yRegion2, mcnt1, mcnt2,
			mcnt3;

	double cSum, cMean, cMedian, cMode, cLfwhm;
	double *im, tdat, idat, ndat, diff, bg;
	//int vsize=(nCompKer+nbg_vec)*SIZE;

	CUDA_SAFE_CALL(
			cudaMemcpy(d_sflag, mystamp, sizeof(int) * flag,
					cudaMemcpyHostToDevice));

	float *d_temp;
	CUDA_SAFE_CALL(cudaMalloc(&d_temp, sizeof(float) * nStamps * SIZE));
	CUDA_SAFE_CALL(cudaMemset(d_temp, 0, sizeof(float) * nStamps * SIZE));
	CUDA_SAFE_CALL(
			cudaMemcpy(d_ksol, kernelSol, sizeof(double) * (mat_size + 1),
					cudaMemcpyHostToDevice));
	/* temp contains the convolved image from fit, fwKSStamp x fwKSStamp */
	gmake_model<<<flag, fwKSStamp*fwKSStamp, fwKSStamp*fwKSStamp*sizeof(double)>>> ( vectors, d_fx, d_fy, d_ksol, d_temp , d_sflag, nCompKer, kerOrder, mat_size, vsize * SIZE);

	float * whole_temp = (float*) malloc(sizeof(float) * nStamps * SIZE);
	CUDA_SAFE_CALL(
			cudaMemcpy(whole_temp, d_temp, sizeof(float) * nStamps * SIZE,
					cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_temp));
	float *temp = (float*) malloc(sizeof(float) * fwKSStamp * fwKSStamp);

	mcnt1 = 0;
	mcnt2 = 0;
	mcnt3 = 0;

	int is;
	double sig1, sig2, sig3;
	for (int n = 0; n < flag; n++) {
		is = mystamp[n];
		xRegion = xstamp[n];
		yRegion = ystamp[n];

		bg = get_background(xRegion, yRegion, kernelSol);

		for (i = 0; i < fwKSStamp * fwKSStamp; i++)
			temp[i] = whole_temp[i + n * SIZE];

		im = stamp[is].krefArea;

		/* get sigma of stamp diff */
		nsig = 0;
		sig1 = 0;
		sig2 = 0;
		sig3 = 0;

		for (j = 0; j < fwKSStamp; j++) {
			yRegion2 = yRegion - hwKSStamp + j;

			for (i = 0; i < fwKSStamp; i++) {
				xRegion2 = xRegion - hwKSStamp + i;

				idx = i + j * fwKSStamp;

				tdat = temp[idx];
				idat = im[idx];
				ndat = imNoise[xRegion2 + rPixX * yRegion2];

				diff = tdat - idat + bg;

				if ((mRData[xRegion2 + rPixX * yRegion2] & FLAG_INPUT_ISBAD)
						|| (fabs(idat) <= ZEROVAL)) {
					continue;
				} else {
					temp[idx] = diff;
				}

				/* check for NaN */
				if ((tdat * 0.0 != 0.0) || (idat * 0.0 != 0.0)) {
					mRData[xRegion2 + rPixX * yRegion2] |= (FLAG_INPUT_ISBAD
							| FLAG_ISNAN);
					continue;
				}

				nsig++;
				sig1 += diff * diff / ndat;
				/*fprintf(stderr, "OK %d %d : %f %f %f\n", xRegion2, yRegion2, tdat, idat, ndat);*/
			}
		}

		if (nsig > 0) {
			sig1 /= nsig;
			if (sig1 >= MAXVAL)
				sig1 = -1;
		} else
			sig1 = -1;

		/* don't do think unless you need to! */
		if (strncmp(figMerit, "v", 1) != 0) {
			if (getStampStats3(temp, xRegion - hwKSStamp, yRegion - hwKSStamp,
					fwKSStamp, fwKSStamp, &cSum, &cMean, &cMedian, &cMode,
					&sig2, &sig3, &cLfwhm, 0x0, 0xffff, 5)) {
				sig2 = -1;
				sig3 = -1;
			} else if (sig2 < 0 || sig2 >= MAXVAL)
				sig2 = -1;
			else if (sig3 < 0 || sig3 >= MAXVAL)
				sig3 = -1;
		}
		if ((sig1 != -1) && (sig1 <= MAXVAL)) {
			m1[mcnt1++] = sig1;
		}
		if ((sig2 != -1) && (sig2 <= MAXVAL)) {
			m2[mcnt2++] = sig2;
		}
		if ((sig3 != -1) && (sig3 <= MAXVAL)) {
			m3[mcnt3++] = sig3;
		}
	}
	*mm1 = mcnt1;
	*mm2 = mcnt2;
	*mm3 = mcnt3;

	free(whole_temp);
	free(temp);
	return;
}

__global__ void build_wxy(double * d_fx, double * d_fy, double* wxy,
		int kerOrder, int ncomp2, float rPixX2, float rPixY2) {
	const int i = threadIdx.x;
	int k, ideg1, ideg2;
	double a1, a2, fx, fy;
	fx = d_fx[i];
	fy = d_fy[i];
	k = 0;
	a1 = 1.0;
	for (ideg1 = 0; ideg1 <= kerOrder; ideg1++) {
		a2 = 1.0;
		for (ideg2 = 0; ideg2 <= kerOrder - ideg1; ideg2++) {
			wxy[i * ncomp2 + k] = a1 * a2;
			a2 *= fy;
			k++;
			__syncthreads();
		}
		a1 *= fx;
	}
}

__global__ void gbuild_scprod(double* wxy, double* vec, float * imRef,
		double * scprod, double* kernelSol, int *xstamp, int *ystamp,
		int *goodstamps, int flag, int vsize, int rPixX, int hwKSStamp,
		int ncomp1, int ncomp2, int fwKSStamp, int msize) {
	short int s, is, i1, i2, ivecbg;

	ivecbg = threadIdx.x + ncomp1 - ncomp1 * ncomp2;
	i1 = (threadIdx.x - 2) / ncomp2;
	i2 = (threadIdx.x - 2) % ncomp2;
	for (s = 0; s < flag; s++) {
		is = goodstamps[s];
		__syncthreads();
		if (threadIdx.x == 1) {  //1        
			kernelSol[threadIdx.x] += scprod[1 + is * msize];
			__syncthreads();
		}
		if (threadIdx.x > 1 && threadIdx.x < ncomp1 * ncomp2 + 2) { //2			
			kernelSol[threadIdx.x] += scprod[i1 + 2 + is * msize]
					* wxy[s * ncomp2 + i2];
			__syncthreads();
		}
		if (threadIdx.x > ncomp1 * ncomp2 + 1) {  //3

			kernelSol[threadIdx.x] += scprod[ivecbg + is * msize];
		}
		__syncthreads();
	}
}

__global__ void gbuild_matrix(double* wxy, double* matrix0, double* matrix,
		int matsize, int *goodstamps, int flag, int vsize, int pixStamp,
		int ncomp1, int ncomp2, int msize, int fwKSStamp) {
	short int ivecbg, i1, i2, j1, j2, is, s;
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	ivecbg = i + ncomp1 - ncomp1 * ncomp2;
	i1 = (i - 2) / ncomp2;
	j1 = (j - 2) / ncomp2;
	i2 = (i - 2) % ncomp2;
	j2 = (j - 2) % ncomp2;
	__syncthreads();
	for (s = 0; s < flag; s++) {
		is = goodstamps[s];
		__syncthreads();
		if (i > 1 && i < (ncomp1 * ncomp2 + 2) && j > 1 && j <= i) {  //1

			matrix[i + j * matsize] += wxy[s * ncomp2 + i2]
					* wxy[s * ncomp2 + j2]
					* matrix0[is * msize * msize + (i1 + 2) * msize + j1 + 2];
			matrix[j + i * matsize] += wxy[s * ncomp2 + i2]
					* wxy[s * ncomp2 + j2]
					* matrix0[is * msize * msize + (i1 + 2) * msize + j1 + 2];

		}
		if (i == 1 && j == 1) {  //2
			matrix[i + j * matsize] += matrix0[is * msize * msize + msize + 1];
			matrix[i + j * matsize] += matrix0[is * msize * msize + msize + 1];

		}
		if (i > 1 && i < (ncomp1 * ncomp2 + 2) && j == 1) {  //3

			matrix[i + j * matsize] += wxy[s * ncomp2 + i2]
					* matrix0[is * msize * msize + (i1 + 2) * msize + 1];
			matrix[j + i * matsize] += wxy[s * ncomp2 + i2]
					* matrix0[is * msize * msize + (i1 + 2) * msize + 1];

		}
		if (i > (ncomp1 * ncomp2 + 1) && i < matsize && j > 1
				&& j < (ncomp1 * ncomp2 + 2)) {  //4

			matrix[i + j * matsize] += matrix0[is * msize * msize
					+ ivecbg * msize + j1 + 2] * wxy[s * ncomp2 + j2];
			matrix[j + i * matsize] += matrix0[is * msize * msize
					+ ivecbg * msize + j1 + 2] * wxy[s * ncomp2 + j2];

		}
		if (i > (ncomp1 * ncomp2 + 1) && i < matsize && j == 1) {  //5
			matrix[i + j * matsize] += matrix0[is * msize * msize
					+ ivecbg * msize + 1];
			matrix[j + i * matsize] += matrix0[is * msize * msize
					+ ivecbg * msize + 1];

		}
		if (i > (ncomp1 * ncomp2 + 1) && i < matsize
				&& j > (ncomp1 * ncomp2 + 1) && j <= i) {  //6

			matrix[i + j * matsize] += matrix0[is * msize * msize
					+ ivecbg * msize + (j + ncomp1 - ncomp1 * ncomp2)];
			matrix[j + i * matsize] += matrix0[is * msize * msize
					+ ivecbg * msize + (j + ncomp1 - ncomp1 * ncomp2)];

		}
		__syncthreads();
	}
}

__global__ void build_matrix_fit(double *d_matrix, int msize) {
	const int i = threadIdx.x;
	d_matrix[i + msize * i] = d_matrix[i + msize * i] * 0.5;
}

__inline__ void build_matrix_first(stamp_struct *stamps, float *imRef,
		double *vectors, double *mat, double *scprod, int *goodstamps,
		int *xstamp, int *ystamp, int flag) {
	/*****************************************************
	 * Build overall matrix including spatial variations
	 *****************************************************/
	int i;
	if (verbose >= 2)
		fprintf(stderr, " Mat_size: %i ncomp2: %i ncomp1: %i nbg_vec: %i \n",
				mat_size, ncomp2, ncomp1, nbg_vec);

	for (i = 0; i < flag; i++) {
		fx[i] = (xstamp[i] - rPixX2) / rPixX2;
		fy[i] = (ystamp[i] - rPixY2) / rPixY2;
	}

	CUDA_SAFE_CALL(
			cudaMemcpy(d_fx, fx, sizeof(double) * flag,
					cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(
			cudaMemcpy(d_fy, fy, sizeof(double) * flag,
					cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(
			cudaMemcpy(d_xstamp, xstamp, sizeof(int) * flag,
					cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(
			cudaMemcpy(d_ystamp, ystamp, sizeof(int) * flag,
					cudaMemcpyHostToDevice));

	dim3 mblock(32, 4);
	dim3 mgrid((mat_size + 1) / 32 + 1, (mat_size + 1) / 4 + 1);
	dim3 mblocks(32, (mat_size + 1) / 32 + 1);
	build_wxy<<<1,flag,0>>>( d_fx, d_fy, d_wxy, kerOrder, ncomp2, rPixX2, rPixY2);

	CUDA_SAFE_CALL(
			cudaMemset(d_matrix, 0,
					sizeof(double) * (mat_size + 1) * (mat_size + 1)));
	CUDA_SAFE_CALL(cudaMemset(d_ksol, 0, sizeof(double) * (nCompTotal + 1)));

	gbuild_matrix<<<mgrid,mblock>>>( d_wxy, mat, d_matrix, mat_size+1, goodstamps, flag, vsize, pixStamp, ncomp1, ncomp2, mats, fwKSStamp);

	build_matrix_fit<<<1,(mat_size+1)>>>(d_matrix, mat_size+1);

	gbuild_scprod<<<1,ncomp+nbg_vec+2>>>( d_wxy, vectors, imRef, scprod, d_ksol, d_xstamp, d_ystamp, goodstamps, flag, vsize, rPixX, hwKSStamp, ncomp1, ncomp2, fwKSStamp, mats);

}

__global__ void one_conv_stamp1(float *dtemp, double *filter_y, float *image,
	int xi, int yi, int hwKSStamp, int hwKernel, int fwKSStamp, int fwKernel,
	int rPixX) {

float imc;
int yc, n;
imc = 0.0;
n = blockIdx.x * 0.25;
__syncthreads();
if (blockIdx.x % 4 == 0) {
	for (yc = -hwKernel; yc <= hwKernel; yc++) {
		imc += image[threadIdx.x + xi - hwKSStamp - hwKernel
				+ rPixX * (threadIdx.y + yi + yc - hwKSStamp)]
				* filter_y[hwKernel - yc + n * fwKernel];

	}

	__syncthreads();
	dtemp[n * 2 * SIZE + threadIdx.x + threadIdx.y * (fwKSStamp + fwKernel)] =
			imc;
}
if (blockIdx.x % 4 == 1) {
	for (yc = -hwKernel; yc <= hwKernel; yc++) {
		imc += image[threadIdx.x + blockDim.x + xi - hwKSStamp - hwKernel
				+ rPixX * (threadIdx.y + yi + yc - hwKSStamp)]
				* filter_y[hwKernel - yc + n * fwKernel];
	}
	__syncthreads();

	dtemp[n * 2 * SIZE + threadIdx.x + blockDim.x
			+ threadIdx.y * (fwKSStamp + fwKernel)] = imc;
}

if (blockIdx.x % 4 == 2) {
	for (yc = -hwKernel; yc <= hwKernel; yc++) {
		imc += image[threadIdx.x + blockDim.x * 2 + xi - hwKSStamp - hwKernel
				+ rPixX * (threadIdx.y + yi + yc - hwKSStamp)]
				* filter_y[hwKernel - yc + n * fwKernel];
	}
	__syncthreads();

	dtemp[n * 2 * SIZE + threadIdx.x + blockDim.x * 2
			+ threadIdx.y * (fwKSStamp + fwKernel)] = imc;
}

if (blockIdx.x % 4 == 3) {
	for (yc = -hwKernel; yc <= hwKernel; yc++) {
		imc += image[threadIdx.x + blockDim.x * 3 + xi - hwKSStamp - hwKernel
				+ rPixX * (threadIdx.y + yi + yc - hwKSStamp)]
				* filter_y[hwKernel - yc + n * fwKernel];
	}
	__syncthreads();

	dtemp[n * 2 * SIZE + threadIdx.x + blockDim.x * 3
			+ threadIdx.y * (fwKSStamp + fwKernel)] = imc;
}
}

__global__ void one_conv_stamp2(double *vectors, float *dtemp, double *filter_x,
	int hwKernel, int fwKSStamp, int fwKernel, int versize, int istamp) {

double imc;
const int n = blockIdx.x + istamp * versize;

int xc;
imc = 0.0;

for (xc = -hwKernel; xc <= hwKernel; xc++) {

	imc += dtemp[threadIdx.x + xc + hwKernel
			+ threadIdx.y * (fwKSStamp + fwKernel) + blockIdx.x * 2 * SIZE]
			* filter_x[hwKernel - xc + (blockIdx.x) * fwKernel];

}

__syncthreads();

vectors[threadIdx.x + threadIdx.y * fwKSStamp + n * SIZE] = imc;

}

__global__ void one_vector_fix(int *ren, double *vectors, int fwKSStamp,
	int versize, int istamp) {
    int m = blockIdx.x * 0.25;
    int l = (blockIdx.x % 4) * SIZE / 4;
    double q;
    if (ren[m] == 1) {
    	q = vectors[threadIdx.x + l + (m + istamp * versize) * SIZE]
			- vectors[threadIdx.x + l + istamp * versize * SIZE];
    	vectors[threadIdx.x + l + (m + istamp * versize) * SIZE] = q;
    }
}

__global__ void one_fill_vec(double *vectors, int xi, int yi, float rPixX2,
	float rPixY2, int nCompKer, int fwKSStamp, int hwKSStamp, int vsize,
	int bgOrder, int istamp) {
	double xf = (float) (xi + threadIdx.x - hwKSStamp - rPixX2) / rPixX2;
	double yf = (float) (yi + blockIdx.x - hwKSStamp - rPixY2) / rPixY2;
	int i, j;
	int nv = nCompKer;
	double ax, ay;
	ax = 1.0;
	for (i = 0; i <= bgOrder; i++) {
		ay = 1.0;
		for (j = 0; j <= bgOrder - i; j++) {
			vectors[threadIdx.x + blockIdx.x * fwKSStamp + nv * SIZE
				+ istamp * vsize * SIZE] = ax * ay;
			ay *= yf;
			++nv;
		}
		ax *= xf;
	}
}

__global__ void one_build_matrix0(double *mat, double *vectors, int vsize,
	int half, int pixStamp, int msize, int istamp) {
/*****************************************************
 * Build least squares matrix for each stamp
 *****************************************************/
	int fwx;
	int k, m;
	double q = 0.0;
	double *vec = (double*) array;

	if (blockIdx.x % 4 == 0) {
		for (m = 0; m <= pixStamp / half; m++) {
			fwx = m * half + threadIdx.y;
			vec[threadIdx.y * vsize + threadIdx.x] = vectors[threadIdx.x * SIZE
				+ fwx + istamp * SIZE * vsize];
			__syncthreads();

			for (k = 0; k < half; k++)
				if (m * half + k < pixStamp)
					q += vec[k * vsize + threadIdx.x]
						* vec[k * vsize + threadIdx.y];
			__syncthreads();
		}
		mat[threadIdx.x + 1 + msize * (threadIdx.y + 1) + istamp * msize * msize] =
			q;
		__syncthreads();
	}

	if (blockIdx.x % 4 == 1) {
		for (m = 0; m <= pixStamp / half; m++) {
			fwx = m * 26 + threadIdx.y;
			vec[threadIdx.y * vsize + threadIdx.x] = vectors[threadIdx.x * SIZE
				+ fwx + istamp * SIZE * vsize];
			vec[threadIdx.y * vsize + threadIdx.x + half] = vectors[fwx
				+ SIZE * (threadIdx.x + half) + istamp * SIZE * vsize];
			__syncthreads();

			for (k = 0; k < half; k++)
				if (m * half + k < pixStamp)
					q += vec[k * vsize + threadIdx.x + half]
						* vec[k * vsize + threadIdx.y];
			__syncthreads();
		}	
		mat[threadIdx.x + half + 1 + msize * (threadIdx.y + 1)
			+ istamp * msize * msize] = q;
		__syncthreads();
	}

	if (blockIdx.x % 4 == 2) {
		for (m = 0; m <= pixStamp / half; m++) {
			fwx = m * half + threadIdx.y;
			vec[threadIdx.y * vsize + threadIdx.x] = vectors[threadIdx.x * SIZE
				+ fwx + istamp * SIZE * vsize];
			vec[threadIdx.y * vsize + threadIdx.x + half] = vectors[fwx
				+ SIZE * (threadIdx.x + half) + istamp * SIZE * vsize];
			__syncthreads();

			for (k = 0; k < half; k++)
				if (m * half + k < pixStamp)
					q += vec[k * vsize + threadIdx.x]
						* vec[k * vsize + threadIdx.y + half];
			__syncthreads();
		}
		mat[(threadIdx.x + 1) + msize * (threadIdx.y + half + 1)
			+ istamp * msize * msize] = q;
		__syncthreads();
	}

	if (blockIdx.x % 4 == 3) {
		for (m = 0; m <= pixStamp / half; m++) {
			fwx = m * half + threadIdx.y;
			vec[threadIdx.y * vsize + threadIdx.x] = vectors[fwx
				+ SIZE * (threadIdx.x + half) + istamp * SIZE * vsize];
			__syncthreads();

			for (k = 0; k < half; k++)
				if (m * half + k < pixStamp)
					q += vec[k * vsize + threadIdx.x]
						* vec[k * vsize + threadIdx.y];
			__syncthreads();
		}
		mat[(threadIdx.x + half + 1) + msize * (threadIdx.y + half + 1)
			+ istamp * msize * msize] = q;
		__syncthreads();
	}
}

__global__ void one_build_scprod0(double *scprod, double *vec, float *image,
	int xi, int yi, int fwKSStamp, int hwKSStamp, int rPixX, int nvec, int nC,
	int istamp) {
/*****************************************************
 * Build the right side of each stamp's least squares matrix
 *    stamp.scprod = degree of kernel fit + 1 bg term
 *****************************************************/
	const int i = threadIdx.x;
	double p0 = 0.0;
	int xc, yc, k;
/* Do eqn 4. in Alard */
	for (xc = -hwKSStamp; xc <= hwKSStamp; xc++) {
		for (yc = -hwKSStamp; yc <= hwKSStamp; yc++) {
			k = xc + hwKSStamp + fwKSStamp * (yc + hwKSStamp);
			p0 += vec[i * SIZE + k + istamp * SIZE * nvec]
				* (double) image[xc + xi + rPixX * (yc + yi)];
		}
	}

	scprod[i + 1 + istamp * mats] = p0;
}

void getStampSig_all(stamp_struct *stamp, double *vectors, double *kernelSol,
	int *xstamp, int *ystamp, int *mystamp, int flag, float *imNoise,
	double *sig1, double *sig2, double *sig3) {

int i, j, idx, nsig, xRegion, yRegion, xRegion2, yRegion2;

double cSum, cMean, cMedian, cMode, cLfwhm;
double tdat, idat, ndat, diff, bg;

float *d_temp;
CUDA_SAFE_CALL(cudaMalloc(&d_temp, sizeof(float) * nStamps * SIZE));
CUDA_SAFE_CALL(cudaMemset(d_temp, 0, sizeof(float) * nStamps * SIZE));
CUDA_SAFE_CALL(
		cudaMemcpy(d_sflag, mystamp, sizeof(int) * flag,
				cudaMemcpyHostToDevice));
CUDA_SAFE_CALL(
		cudaMemcpy(d_ksol, kernelSol, sizeof(double) * (mat_size + 1),
				cudaMemcpyHostToDevice));
/* temp contains the convolved image from fit, fwKSStamp x fwKSStamp */
gmake_model<<<flag, fwKSStamp*fwKSStamp, fwKSStamp*fwKSStamp*sizeof(double)>>> ( vectors, d_fx, d_fy, d_ksol , d_temp , d_sflag, nCompKer, kerOrder, mat_size, vsize * SIZE);

float * whole_temp = (float*) malloc(sizeof(float) * nStamps * SIZE);
CUDA_SAFE_CALL(
		cudaMemcpy(whole_temp, d_temp, sizeof(float) * nStamps * SIZE,
				cudaMemcpyDeviceToHost));
CUDA_SAFE_CALL(cudaFree(d_temp));
float *temp = (float*) malloc(sizeof(float) * fwKSStamp * fwKSStamp);

int is;
#pragma omp parallel for private(  i, j, idx,  nsig, xRegion, yRegion, xRegion2, yRegion2, tdat, idat, ndat, diff, bg ,is) 
for (int n = 0; n < flag; n++) {
	is = mystamp[n];
	xRegion = xstamp[n];
	yRegion = ystamp[n];

	bg = get_background(xRegion, yRegion, kernelSol);

	/* get sigma of stamp diff */
	nsig = 0;
	sig1[is] = 0;
	sig2[is] = 0;
	sig3[is] = 0;

	for (j = 0; j < fwKSStamp; j++) {
		yRegion2 = yRegion - hwKSStamp + j;

		for (i = 0; i < fwKSStamp; i++) {
			xRegion2 = xRegion - hwKSStamp + i;

			idx = i + j * fwKSStamp;

			tdat = whole_temp[n * SIZE + idx];
			idat = stamp[is].krefArea[idx];
			ndat = imNoise[xRegion2 + rPixX * yRegion2];

			diff = tdat - idat + bg;

			if ((mRData[xRegion2 + rPixX * yRegion2] & FLAG_INPUT_ISBAD)
					|| (fabs(idat) <= ZEROVAL)) {
				continue;
			} else {
				whole_temp[n * SIZE + idx] = diff;
			}

			/* check for NaN */
			if ((tdat * 0.0 != 0.0) || (idat * 0.0 != 0.0)) {
				mRData[xRegion2 + rPixX * yRegion2] |= (FLAG_INPUT_ISBAD
						| FLAG_ISNAN);
				continue;
			}

			nsig++;
			sig1[is] += diff * diff / ndat;
			/*fprintf(stderr, "OK %d %d : %f %f %f\n", xRegion2, yRegion2, tdat, idat, ndat);*/
		}
	}
	if (nsig > 0) {
		sig1[is] /= nsig;
		if (sig1[is] >= MAXVAL)
			sig1[is] = -1;
	} else
		sig1[is] = -1;

	/* don't do think unless you need to! */
	if (strncmp(figMerit, "v", 1) != 0) {
		if (getStampStats3(whole_temp + n * SIZE, xRegion - hwKSStamp,
				yRegion - hwKSStamp, fwKSStamp, fwKSStamp, &cSum, &cMean,
				&cMedian, &cMode, &(sig2[is]), &(sig3[is]), &cLfwhm, 0x0,
				0xffff, 5)) {
			sig2[is] = -1;
			sig3[is] = -1;
		} else if (sig2[is] < 0 || sig2[is] >= MAXVAL)
			sig2[is] = -1;
		else if (sig3[is] < 0 || sig3[is] >= MAXVAL)
			sig3[is] = -1;
	}

}
free(whole_temp);
free(temp);
return;
}

int fillStamp_one(stamp_struct *stamp, double *vectors, double *mat,
	double *scprod, float *imConv, float *imRef, float *image, int istamp) {
/*****************************************************
 * Fills stamp->vectors with convolved images, and 
 *   pixel indices multiplied by each other for background fit 
 *****************************************************/

float * dtemp;

if (verbose >= 1)
	fprintf(stderr, "    xs  : %4i ys  : %4i sig: %6.3f sscnt: %4i nss: %4i \n",
			stamp[istamp].x, stamp[istamp].y, stamp[istamp].chi2,
			stamp[istamp].sscnt, stamp[istamp].nss);

if (stamp[istamp].sscnt >= stamp[istamp].nss) {
	/* have gone through all the good substamps, reject this stamp */
	/*if (verbose >= 2) fprintf(stderr, "    ******** REJECT stamp (out of substamps)\n");*/
	if (verbose >= 1)
		fprintf(stderr, "        Reject stamp\n");
	return 1;
}
if (cutSStamp(&stamp[istamp], image))
	return 1;
int xs = stamp[istamp].xss[stamp[istamp].sscnt];
int ys = stamp[istamp].yss[stamp[istamp].sscnt];
/* stores kernel weight mask for each order */
CUDA_SAFE_CALL(cudaMalloc(&dtemp, sizeof(float) * nCompKer * SIZE * 2));
CUDA_SAFE_CALL(cudaMemset(dtemp, 0, sizeof(float) * nCompKer * SIZE * 2));
CUDA_SAFE_CALL(
		cudaMemset(vectors + istamp * vsize * SIZE, 0,
				sizeof(double) * vsize * SIZE));
CUDA_SAFE_CALL(
		cudaMemset(mat + istamp * mats * mats, 0,
				sizeof(double) * mats * mats));
CUDA_SAFE_CALL(cudaMemset(scprod + istamp * mats, 0, sizeof(double) * mats));

dim3 thread1((fwKSStamp + fwKernel) * 0.25, fwKSStamp);
dim3 block1(nCompKer * 4);
dim3 thread2(fwKSStamp, fwKSStamp);
dim3 blocks(nCompKer);
dim3 thread3(vsize / 2, vsize / 2);
dim3 block3(fwKSStamp, 1);
one_conv_stamp1<<< nCompKer*4 , thread1>>>( dtemp, dfilter_y, imConv, xs,ys, hwKSStamp, hwKernel, fwKSStamp, fwKernel,rPixX);

one_conv_stamp2<<< nCompKer ,thread2 >>>(vectors, dtemp, dfilter_x, hwKernel, fwKSStamp, fwKernel ,vsize, istamp);

CUDA_SAFE_CALL(cudaFree(dtemp));

one_vector_fix<<<block1 , SIZE * 0.25>>>( dren ,vectors, fwKSStamp, vsize, istamp);

/* get the krefArea data */
/* fill stamp->vectors[nvec+++] with x^(bg) * y^(bg) for background fit*/
one_fill_vec<<< block3, fwKSStamp >>>(vectors, xs,ys, rPixX2, rPixY2, nCompKer, fwKSStamp, hwKSStamp, vsize, bgOrder, istamp);

CUDA_SAFE_CALL (cudaThreadSynchronize());

/* build stamp->mat from stamp->vectors*/
one_build_matrix0<<<4,thread3,sizeof(double)*vsize*vsize/2>>>( mat, vectors , vsize , vsize/2 , pixStamp, mats, istamp );

/* build stamp->scprod from stamp->vectors and imRef  */

one_build_scprod0<<<1,vsize>>>(scprod, vectors, imRef, xs, ys, fwKSStamp, hwKSStamp, rPixX, vsize ,nC, istamp);

return 0;
}

char check_again(stamp_struct *stamps, double *vectors, double *mat,
	double *scprod, double *kernelSol, float *imConv, float *imRef,
	float *image, float *imNoise, int *mystamp, int *xstamp, int *ystamp,
	int flag, double *meansigSubstamps, double *scatterSubstamps,
	int *NskippedSubstamps) {
/*****************************************************
 * Check for bad stamps after the global fit - iterate if necessary
 *****************************************************/

int istamp, nss, scnt;
double sig, mean, stdev;
char check;

float *ss;

ss = (float *) calloc(nStamps, sizeof(float));
nss = 0;

sig = 0;
check = 0;
mean = stdev = 0.0;
*NskippedSubstamps = 0;
double *sig1, *sig2, *sig3;
CUDA_SAFE_CALL(cudaMallocHost(&sig1, sizeof(double) * nStamps));
CUDA_SAFE_CALL(cudaMallocHost(&sig2, sizeof(double) * nStamps));
CUDA_SAFE_CALL(cudaMallocHost(&sig3, sizeof(double) * nStamps));

getStampSig_all(stamps, vectors, kernelSol, xstamp, ystamp, mystamp, flag,
		imNoise, sig1, sig2, sig3);

//flag=0;
for (istamp = 0; istamp < nStamps; istamp++) {

	/* if was fit with a good legit substamp */
	if (stamps[istamp].sscnt < stamps[istamp].nss) {

		if ((strncmp(figMerit, "v", 1) == 0 && (sig1[istamp] == -1))
				|| (strncmp(figMerit, "s", 1) == 0 && (sig2[istamp] == -1))
				|| (strncmp(figMerit, "h", 1) == 0 && (sig3[istamp] == -1))) {

			/* something went wrong with this one... */
			if (verbose >= 2)
				fprintf(stderr,
						"\n    # %d    xss: %4i yss: %4i sig: %6.3f sscnt: %2i nss: %2i ITERATE substamp (BAD)\n",
						istamp, stamps[istamp].xss[stamps[istamp].sscnt],
						stamps[istamp].yss[stamps[istamp].sscnt], sig,
						stamps[istamp].sscnt, stamps[istamp].nss);

			stamps[istamp].sscnt++;
			fillStamp_one(stamps, vectors, mat, scprod, imConv, imRef, image,
					istamp);

			check = 1;

		} else {
			if (strncmp(figMerit, "v", 1) == 0)
				sig = sig1[istamp];
			else if (strncmp(figMerit, "s", 1) == 0)
				sig = sig2[istamp];
			else if (strncmp(figMerit, "h", 1) == 0)
				sig = sig3[istamp];

			if (verbose >= 2)
				fprintf(stderr,
						"    # %d    xss: %4i yss: %4i sig: %6.3f sscnt: %2i nss: %2i OK\n",
						istamp, stamps[istamp].xss[stamps[istamp].sscnt],
						stamps[istamp].yss[stamps[istamp].sscnt], sig,
						stamps[istamp].sscnt, stamps[istamp].nss);

			stamps[istamp].chi2 = sig;
			ss[nss++] = sig;

		}
	} else {
		(*NskippedSubstamps)++;
		if (verbose >= 2)
			fprintf(stderr, "    xs : %4i ys : %4i skipping... \n",
					stamps[istamp].x, stamps[istamp].y);
	}
}

sigma_clip(ss, nss, &mean, &stdev, 10);
fprintf(stderr, "    Mean sig: %6.3f stdev: %6.3f\n", mean, stdev);
fprintf(stderr, "    Iterating through stamps with sig > %.3f\n",
		mean + kerSigReject * stdev);

/* save the mean and scatter so that it can be saved in the fits header */
(*meansigSubstamps) = mean;
(*scatterSubstamps) = stdev;

scnt = 0;
for (istamp = 0; istamp < nStamps; istamp++) {
	/* if currently represented by a good substamp */
	if (stamps[istamp].sscnt < stamps[istamp].nss) {

		/* no fabs() here, keep good stamps kerSigReject on the low side! */
		if ((stamps[istamp].chi2 - mean) > kerSigReject * stdev) {
			if (verbose >= 2)
				fprintf(stderr,
						"    # %d    xss: %4i yss: %4i sig: %6.3f sscnt: %2i nss: %2i ITERATE substamp (poor sig)\n",
						istamp, stamps[istamp].xss[stamps[istamp].sscnt],
						stamps[istamp].yss[stamps[istamp].sscnt],
						stamps[istamp].chi2, stamps[istamp].sscnt,
						stamps[istamp].nss);

			stamps[istamp].sscnt++;
			scnt += (!(fillStamp_one(stamps, vectors, mat, scprod, imConv,
					imRef, image, istamp)));

			if (verbose >= 2)
				fprintf(stderr, "\n");

			check = 1;
		} else
			scnt += 1;
	}
}

fprintf(stderr, "    %d out of %d stamps remain\n", scnt, nStamps);

CUDA_SAFE_CALL(cudaFreeHost(sig1));
CUDA_SAFE_CALL(cudaFreeHost(sig2));
CUDA_SAFE_CALL(cudaFreeHost(sig3));
free(ss);
return check;
}

extern "C" void fitKernel(stamp_struct *stamps, double *vectors, double *mat,
	double *scprod, float *imRef, float *imConv, float *image, float *imNoise,
	double *kernelSol, double *meansigSubstamps, double *scatterSubstamps,
	int *NskippedSubstamps) {
/*****************************************************
 * Complete fit for kernel solution
 *****************************************************/

char check;
int istamp, flag;
flag = 0;

if (verbose >= 2)
	fprintf(stderr, " Mat_size: %i ncomp2: %i ncomp1: %i nbg_vec: %i \n",
			mat_size, ncomp2, ncomp1, nbg_vec);

CUDA_SAFE_CALL(
		cudaMemset(d_matrix, 0,
				sizeof(double) * (mat_size + 1) * (mat_size + 1)));
CUDA_SAFE_CALL(cudaMemset(d_ksol, 0, sizeof(double) * (nCompTotal + 1)));

for (istamp = 0; istamp < nStamps; istamp++) {
	if (stamps[istamp].sscnt < stamps[istamp].nss) {
		mystamp[flag] = istamp;

		xstamp[flag] = stamps[istamp].xss[stamps[istamp].sscnt];
		ystamp[flag] = stamps[istamp].yss[stamps[istamp].sscnt];

		flag++;
	}
}
CUDA_SAFE_CALL(
		cudaMemcpy(d_sflag, mystamp, sizeof(int) * flag,
				cudaMemcpyHostToDevice));

build_matrix_first(stamps, imRef, vectors, mat, scprod, d_sflag, xstamp, ystamp,
		flag);
if (verbose >= 2)
	fprintf(stderr, " Expanding Matrix For Full Fit\n");
double *temp_matrix;
CUDA_SAFE_CALL(
		cudaMallocHost(&temp_matrix,
				sizeof(double) * (mat_size + 1) * (mat_size + 1)));

CUDA_SAFE_CALL(
		cudaMemcpy(temp_matrix, d_matrix,
				(mat_size + 1) * sizeof(double) * (mat_size + 1),
				cudaMemcpyDeviceToHost));

double * testKerSol = (double*) malloc(sizeof(double) * (nCompTotal + 1));
CUDA_SAFE_CALL(
		cudaMemcpy(testKerSol, d_ksol, sizeof(double) * (nCompTotal + 1),
				cudaMemcpyDeviceToHost));

ludcmp_d1(temp_matrix, mat_size, mat_size + 1, indx);
lubksb_d1(temp_matrix, mat_size, mat_size + 1, indx, testKerSol);

check = check_again(stamps, vectors, mat, scprod, testKerSol, imConv, imRef,
		image, imNoise, mystamp, xstamp, ystamp, flag, meansigSubstamps,
		scatterSubstamps, NskippedSubstamps);

while (check) {
	flag = 0;
	for (istamp = 0; istamp < nStamps; istamp++) {
		if (stamps[istamp].sscnt < stamps[istamp].nss) {
			mystamp[flag] = istamp;

			xstamp[flag] = stamps[istamp].xss[stamps[istamp].sscnt];
			ystamp[flag] = stamps[istamp].yss[stamps[istamp].sscnt];

			flag++;
		}
	}

	CUDA_SAFE_CALL(
			cudaMemcpy(d_sflag, mystamp, sizeof(int) * nStamps,
					cudaMemcpyHostToDevice));
	fprintf(stderr, "\n Re-Expanding Matrix\n");

	build_matrix_first(stamps, imRef, vectors, mat, scprod, d_sflag, xstamp,
			ystamp, flag);
	CUDA_SAFE_CALL(
			cudaMemcpy(temp_matrix, d_matrix,
					(mat_size + 1) * sizeof(double) * (mat_size + 1),
					cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(
			cudaMemcpy(testKerSol, d_ksol, sizeof(double) * (nCompTotal + 1),
					cudaMemcpyDeviceToHost));

	ludcmp_d1(temp_matrix, mat_size, mat_size + 1, indx);
	lubksb_d1(temp_matrix, mat_size, mat_size + 1, indx, testKerSol);
	fprintf(stderr, " Checking again\n");

	check = check_again(stamps, vectors, mat, scprod, testKerSol, imConv, imRef,
			image, imNoise, mystamp, xstamp, ystamp, flag, meansigSubstamps,
			scatterSubstamps, NskippedSubstamps);
}
fprintf(stderr, " Sigma clipping of bad stamps converged, kernel determined\n");

CUDA_SAFE_CALL(
		cudaMemcpy(kernelSol, testKerSol, sizeof(double) * (nCompTotal + 1),
				cudaMemcpyHostToDevice));
cudaFreeHost(temp_matrix);
free(testKerSol);
return;
}

__global__ void convolve1(float* d_image, double* d_kernel, float* dcrdata,
	int* cMask, int* d_mRData, const int kcStep, const int hwKernel,
	const int xSize, const int ySize, int nsteps_x, const float kerFracMask) {

int tx = threadIdx.x + blockIdx.x * kcStep;
int ty = threadIdx.y + blockIdx.y * kcStep;
__shared__
double subkernel[D_HWKERNEL * 2 + 1][D_HWKERNEL * 2 + 1];
__shared__
float image[2 * (D_HWKERNEL * 2 + 1)][2 * (D_HWKERNEL * 2 + 1)];
int mbit = 0x0;
double aks = 0.0;
double uks = 0.0;
double q = 0;

subkernel[threadIdx.x][threadIdx.y] = d_kernel[(blockIdx.y * nsteps_x
		+ blockIdx.x) * SIZE + threadIdx.x + (2 * hwKernel + 1) * threadIdx.y];

image[threadIdx.x][threadIdx.y] = d_image[tx + xSize * ty];
image[threadIdx.x + kcStep][threadIdx.y] = d_image[tx + kcStep + xSize * ty];
image[threadIdx.x][threadIdx.y + kcStep] = d_image[tx + xSize * (ty + kcStep)];
image[threadIdx.x + kcStep][threadIdx.y + kcStep] = d_image[tx + kcStep
		+ xSize * (ty + kcStep)];
__syncthreads();

if ((tx + 2 * hwKernel) < xSize && (ty + 2 * hwKernel) < ySize) {
	for (int i = 0; i <= 2 * hwKernel; i++) {
		for (int j = 0; j <= 2 * hwKernel; j++) {
			q += image[threadIdx.x + i][threadIdx.y + j]
					* subkernel[2 * hwKernel - i][2 * hwKernel - j];
			mbit |= cMask[tx + i + xSize * (ty + j)];
			aks += fabs(subkernel[2 * hwKernel - i][2 * hwKernel - j]);
			if (!(cMask[tx + i + xSize * (ty + j)] & 0x80)) {
				uks += fabs(subkernel[2 * hwKernel - i][2 * hwKernel - j]);
			}

		}
	}

	__syncthreads();
	dcrdata[tx + hwKernel + xSize * (ty + hwKernel)] = q;
	d_mRData[tx + hwKernel + xSize * (ty + hwKernel)] |= cMask[tx + hwKernel
			+ xSize * (ty + hwKernel)];
	d_mRData[tx + hwKernel + xSize * (ty + hwKernel)] |= 0x8000
			* ((cMask[tx + hwKernel + xSize * (ty + hwKernel)] & 0x80) > 0);
	if (mbit) {
		if ((uks / aks) < kerFracMask) {
			d_mRData[tx + hwKernel + xSize * (ty + hwKernel)] |=
					(0x8000 | 0x10);
		} else {
			d_mRData[tx + hwKernel + xSize * (ty + hwKernel)] |= 0x40;
		}
	}
}

}

__global__ void variance_1(float* d_image, float* d_variance, double* d_kernel,
	float* dcrdata, int* cMask, int* d_mRData, float* d_vData, const int kcStep,
	const int hwKernel, const int xSize, const int ySize, const int nsteps_x,
	const float kerFracMask) {
const int tx = threadIdx.x + blockIdx.x * kcStep;
const int ty = threadIdx.y + blockIdx.y * kcStep;
__shared__
double subkernel[D_HWKERNEL * 2 + 1][D_HWKERNEL * 2 + 1];
__shared__
float image[2 * (D_HWKERNEL * 2 + 1)][2 * (D_HWKERNEL * 2 + 1)];
__shared__
float vari[2 * (D_HWKERNEL * 2 + 1)][2 * (D_HWKERNEL * 2 + 1)];
int mbit = 0x0;
double aks = 0.0;
double uks = 0.0;
double q = 0.0;
double qv = 0.0;

subkernel[threadIdx.x][threadIdx.y] = d_kernel[(blockIdx.y * nsteps_x
		+ blockIdx.x) * SIZE + threadIdx.x + (2 * hwKernel + 1) * threadIdx.y];
__syncthreads();

if ((tx + hwKernel) < xSize && (ty + hwKernel) < ySize) {
	image[threadIdx.x][threadIdx.y] = d_image[tx + xSize * ty];
	image[threadIdx.x + kcStep][threadIdx.y] =
			d_image[tx + kcStep + xSize * ty];
	image[threadIdx.x][threadIdx.y + kcStep] = d_image[tx
			+ xSize * (ty + kcStep)];
	image[threadIdx.x + kcStep][threadIdx.y + kcStep] = d_image[tx + kcStep
			+ xSize * (ty + kcStep)];

	vari[threadIdx.x][threadIdx.y] = d_variance[tx + xSize * ty];
	vari[threadIdx.x + kcStep][threadIdx.y] = d_variance[tx + kcStep
			+ xSize * ty];
	vari[threadIdx.x][threadIdx.y + kcStep] = d_variance[tx
			+ xSize * (ty + kcStep)];
	vari[threadIdx.x + kcStep][threadIdx.y + kcStep] = d_variance[tx + kcStep
			+ xSize * (ty + kcStep)];
	__syncthreads();
}

if ((tx + 2 * hwKernel) < xSize && (ty + 2 * hwKernel) < ySize) {
	for (int i = 0; i <= 2 * hwKernel; i++) {
		for (int j = 0; j <= 2 * hwKernel; j++) {
			qv += vari[threadIdx.x + i][threadIdx.y + j]
					* subkernel[2 * hwKernel - i][2 * hwKernel - j];
			q += image[threadIdx.x + i][threadIdx.y + j]
					* subkernel[2 * hwKernel - i][2 * hwKernel - j];
			mbit |= cMask[tx + i + xSize * (ty + j)];
			aks += fabs(subkernel[2 * hwKernel - i][2 * hwKernel - j]);
			if (!(cMask[tx + i + xSize * (ty + j)] & 0x80)) {
				uks += fabs(subkernel[2 * hwKernel - i][2 * hwKernel - j]);
			}
		}
	}

	__syncthreads();
	dcrdata[tx + hwKernel + xSize * (ty + hwKernel)] = q;
	d_mRData[tx + hwKernel + xSize * (ty + hwKernel)] |= cMask[tx + hwKernel
			+ xSize * (ty + hwKernel)];
	d_mRData[tx + hwKernel + xSize * (ty + hwKernel)] |= 0x8000
			* ((cMask[tx + hwKernel + xSize * (ty + hwKernel)] & 0x80) > 0);
	if (mbit) {
		if ((uks / aks) < kerFracMask) {
			d_mRData[tx + hwKernel + xSize * (ty + hwKernel)] |=
					(0x8000 | 0x10);
		} else {
			d_mRData[tx + hwKernel + xSize * (ty + hwKernel)] |= 0x40;
		}
	}
	d_vData[tx + hwKernel + xSize * (ty + hwKernel)] = qv;

}
}

__global__ void variance_2(double * kernel_vec, float* d_image, float* variance,
	double *d_kercoe, int nCompKer, float* dcrdata, int* d_cMask, int* d_mRData,
	float* d_vData, const int kcStep, const int hwKernel, const int xSize,
	int ySize, const int fwKernel, const int nsteps_x,
	const float kerFracMask) {
int x = threadIdx.x / fwKernel;
int y = threadIdx.x % fwKernel;
int total = x + blockIdx.x * kcStep + xSize * (y + blockIdx.y * kcStep);
int fx = x + y * fwKernel * 2;
int mbit = 0x0;
int temp_mRData = 0x0;
int i, j, temp, square = fwKernel * fwKernel;
double aks = 0.0;
double uks = 0.0;
double q = 0.0;
double qv = 0.0;
double temp_ker;
double *kercoe = (double*) array;
double *subkernel = (double*) &kercoe[nCompKer];
double *double_subker = (double *) &subkernel[fwKernel * fwKernel];
float *image = (float *) &double_subker[fwKernel * fwKernel];
int *cMask = (int *) &image[fwKernel * fwKernel * 4];
float *var = (float *) &cMask[fwKernel * fwKernel * 4];

if (threadIdx.x < nCompKer)
	kercoe[threadIdx.x] = d_kercoe[threadIdx.x
			+ (blockIdx.x + blockIdx.y * gridDim.x) * nCompKer];

__syncthreads();
for (i = 0; i < nCompKer; i++) {

	q += kercoe[i] * kernel_vec[i * SIZE + threadIdx.x];  //make_kernel2
}
subkernel[threadIdx.x] = q;
double_subker[threadIdx.x] = q * q;

if ((x + blockIdx.x * kcStep) < xSize) { //initialize
	temp = total + kcStep * xSize;
	image[fx] = d_image[total];
	image[fx + 2 * fwKernel * kcStep] = d_image[temp];

	var[fx] = variance[total];
	var[fx + 2 * fwKernel * kcStep] = variance[temp];

	cMask[fx] = d_cMask[total];
	cMask[fx + 2 * fwKernel * kcStep] = d_cMask[temp];
}

if ((x + blockIdx.x * kcStep) < xSize) { //initialize
	temp = total + kcStep * xSize + kcStep;
	image[fx + kcStep] = d_image[total + kcStep];
	image[fx + kcStep + 2 * fwKernel * kcStep] = d_image[temp];

	var[fx + kcStep] = variance[total + kcStep];
	var[fx + kcStep + 2 * fwKernel * kcStep] = variance[temp];

	cMask[fx + kcStep] = d_cMask[total + kcStep];
	cMask[fx + kcStep + 2 * fwKernel * kcStep] = d_cMask[temp];
}
q = 0.0;
__syncthreads();

if ((x + blockIdx.x * kcStep + 2 * hwKernel) < xSize) { //compute
	for (i = 0; i <= 2 * hwKernel; i++) {
		temp = square - 1 - i;
		for (j = 0; j <= 2 * hwKernel; j++) {
			temp_ker = subkernel[temp - fwKernel * j];
			q += image[fx + i + j * fwKernel * 2] * temp_ker;
			mbit |= cMask[fx + i + j * fwKernel * 2];
			aks += fabs(temp_ker);
			if (!(cMask[fx + i + j * fwKernel * 2] & 0x80)) {
				uks += fabs(temp_ker);
			}
			qv += var[fx + i + j * fwKernel * 2]
					* double_subker[temp - fwKernel * j];

		}

	}

	//return result
	dcrdata[total + hwKernel + xSize * (hwKernel)] = q;
	temp_mRData |= cMask[fx + hwKernel + 2 * fwKernel * (hwKernel)];
	temp_mRData |= 0x8000
			* ((cMask[fx + hwKernel + 2 * fwKernel * hwKernel] & 0x80) > 0);
	if (mbit) {
		if ((uks / aks) < kerFracMask) {
			temp_mRData |= (0x8000 | 0x10);
		} else {
			temp_mRData |= 0x40;
		}
	}
	d_mRData[total + hwKernel + xSize * (hwKernel)] |= temp_mRData;
	d_vData[total + hwKernel + xSize * (hwKernel)] = qv;

}
}

__global__ void get_back(float *dcrdata, double *kernelSol, const int kcStep,
	const int rPixX, const int rPixY, const int hwKernel, const int ncompBG,
	const int bgOrder) {
const int bx = blockIdx.x * kcStep + threadIdx.x % kcStep + hwKernel;
const int by = blockIdx.y * kcStep + threadIdx.x / kcStep + hwKernel;
double back;
double *kersol = (double *) array;
int i, j, k = 1;
back = 0.0;
if (threadIdx.x < 4)
	kersol[threadIdx.x] = kernelSol[threadIdx.x + ncompBG];

double xf = (bx - 0.5 * rPixX) / (0.5 * rPixX);
double yf = (by - 0.5 * rPixY) / (0.5 * rPixY);

double ax, ay;
__syncthreads();

if ((hwKernel + bx) < rPixX && bx >= hwKernel && by >= hwKernel) {
	ax = 1.0;
	for (i = 0; i <= bgOrder; i++) {
		ay = 1.0;
		for (j = 0; j <= bgOrder - i; j++) {
			back += kersol[k++] * ax * ay;
			ay *= yf;
		}
		ax *= xf;
	}

	dcrdata[bx + rPixX * by] += back;
}
}

__global__ void make_kernel1(double* xf, double* yf, double *kernelSol,
	double *d_kercoe, int kerOrder, int nCompKer, int nsteps_x) {
int k, ix, iy;
double ax, ay;
double *kernel_coeffs = (double *) array;
k = 2 + (threadIdx.x - 1) * (kerOrder + 1) * (kerOrder + 2) / 2;
if (threadIdx.x > 0) {
	kernel_coeffs[threadIdx.x] = 0.0;
	__syncthreads();
	ax = 1.0;
	for (ix = 0; ix <= kerOrder; ix++) {
		ay = 1.0;
		for (iy = 0; iy <= kerOrder - ix; iy++) {
			kernel_coeffs[threadIdx.x] += kernelSol[k++] * ax * ay;
			ay *= yf[blockIdx.y];
		}
		ax *= xf[blockIdx.x];
	}
} else
	kernel_coeffs[threadIdx.x] = kernelSol[1];

d_kercoe[nCompKer * (blockIdx.x + nsteps_x * blockIdx.y) + threadIdx.x] =
		kernel_coeffs[threadIdx.x];
}

__global__ void make_kernel2(double* d_kernel, double * kernel_vec,
	double *d_kercoe, int nCompKer, int pixKernel) {
double *kernel = (double*) array;
double *kercoe = (double*) &kernel[pixKernel]; //test
int i1;
kernel[threadIdx.x] = 0.0;
__syncthreads();

if (threadIdx.x < nCompKer)
	kercoe[threadIdx.x] = d_kercoe[threadIdx.x + blockIdx.x * nCompKer];

__syncthreads();
for (i1 = 0; i1 < nCompKer; i1++) {

	kernel[threadIdx.x] += kercoe[i1] * kernel_vec[i1 * SIZE + threadIdx.x];
}
__syncthreads();
d_kernel[blockIdx.x * SIZE + threadIdx.x] = kernel[threadIdx.x];
}

extern "C" void spatial_convolve(float *d_image, float * image,
	float **variance, int xSize, int ySize, double ** kernel_sol, float *cRdata,
	int *cMask) {
/*****************************************************
 * Take image and convolve it using the kernelSol every kernel width
 *****************************************************/

int l, k, i, j, i2, j2, ni, mbit, jc, jk, ic, ik, nc, i1, j1, nsteps_x,
		nsteps_y, i0, j0, dovar = 1;
float *vData = NULL, *var = (float*) malloc(sizeof(float) * xSize * ySize);
double *d_kersol = *kernel_sol;
double q, qv, kk, aks, uks;

if ((*variance) == NULL)
	dovar = 0;
else
	dovar = 1;

if (dovar) {
	if (!(vData = (float *) malloc(xSize * ySize * sizeof(float)))) {
		return;
	}
}
nsteps_x = ceil((double) (xSize) / (double) kcStep);
nsteps_y = ceil((double) (ySize) / (double) kcStep);

int gpupart = nsteps_x * GPU_PART;

for (j1 = 0; j1 < nsteps_y; j1++) {
	j0 = j1 * kcStep + 2 * hwKernel;
	fy[j1] = (j0 - rPixY2) / rPixY2;
}
for (i1 = 0; i1 < nsteps_x; i1++) {
	i0 = i1 * kcStep + 2 * hwKernel;
	fx[i1] = (i0 - rPixX2) / rPixX2;
}

if (nsteps_x > nStamps) {
	cudaFree (d_fx);
	cudaFree (d_fy);
	CUDA_SAFE_CALL(cudaMalloc(&d_fx, sizeof(double) * nsteps_x));
	CUDA_SAFE_CALL(cudaMalloc(&d_fy, sizeof(double) * nsteps_y));
}

CUDA_SAFE_CALL(
		cudaMemcpy(d_fx, fx, sizeof(double) * nsteps_x,
				cudaMemcpyHostToDevice));

CUDA_SAFE_CALL(
		cudaMemcpy(d_fy, fy, sizeof(double) * nsteps_x,
				cudaMemcpyHostToDevice));

CUDA_SAFE_CALL(
		cudaMalloc((void**) &d_kercoe,
				sizeof(double) * nCompKer * nsteps_x * gpupart));
CUDA_SAFE_CALL(
		cudaMemset(d_kercoe, 0,
				sizeof(double) * nCompKer * nsteps_x * gpupart));

CUDA_SAFE_CALL(
		cudaMalloc((void**) &d_kernel,
				sizeof(double) * SIZE * nsteps_x * gpupart));
CUDA_SAFE_CALL(
		cudaMemset(d_kernel, 0, sizeof(double) * SIZE * nsteps_x * gpupart));

CUDA_SAFE_CALL(cudaMallocHost(kernel_sol, sizeof(double) * (nCompTotal + 1)));
CUDA_SAFE_CALL(
		cudaMemcpy(*kernel_sol, d_kersol, sizeof(double) * (nCompTotal + 1),
				cudaMemcpyDeviceToHost));

float * d_variance, *dcrdata, *d_vData;
int *d_cMask;
int * d_mRdata;

CUDA_SAFE_CALL(
		cudaMalloc(&dcrdata, sizeof(float) * xSize * (gpupart + 1) * kcStep));
CUDA_SAFE_CALL(
		cudaMemset(dcrdata, 0, sizeof(float) * xSize * (gpupart + 1) * kcStep));

for (int i = 0; i < xSize * ySize; i++) {
	var[i] = (*variance)[i];
}

CUDA_SAFE_CALL(cudaMalloc(&d_mRdata, sizeof(int) * xSize * ySize));
CUDA_SAFE_CALL(cudaMemset(d_mRdata, 0, sizeof(int) * xSize * ySize));

CUDA_SAFE_CALL(cudaMalloc(&d_cMask, sizeof(int) * xSize * ySize));
CUDA_SAFE_CALL(
		cudaMemcpy(d_cMask, cMask, sizeof(int) * xSize * ySize,
				cudaMemcpyHostToDevice));
if (dovar) {
	cudaMalloc(&d_vData, sizeof(float) * xSize * ySize);
	cudaMemset(d_vData, 0, sizeof(float) * xSize * ySize);
	cudaMalloc(&d_variance, sizeof(float) * xSize * ySize);
	cudaMemcpy(d_variance, var, sizeof(float) * xSize * ySize,
			cudaMemcpyHostToDevice);
}
dim3 blocks(nsteps_x, nsteps_y);
dim3 blocks_gpu(nsteps_x, gpupart);
dim3 threads(kcStep, kcStep);

make_kernel1<<<blocks_gpu,nCompKer,sizeof(double)*nCompKer>>>(d_fx, d_fy, d_kersol, d_kercoe, kerOrder, nCompKer, nsteps_x);

if (dovar) {

if (convolveVariance)
variance_1<<<blocks,threads,5*sizeof(double)*fwKernel*fwKernel>>>( d_image, d_variance, d_kernel, dcrdata , d_cMask, d_mRdata, d_vData, kcStep, hwKernel, xSize, ySize, nsteps_x, kerFracMask);

else
variance_2<<<blocks_gpu, fwKernel*fwKernel,sizeof(double)*(nCompKer+8*fwKernel*fwKernel)>>>( dkernel_vec, d_image, d_variance, d_kercoe, nCompKer, dcrdata , d_cMask, d_mRdata, d_vData, kcStep, hwKernel, xSize, ySize, fwKernel, nsteps_x, kerFracMask);

} else {

convolve1<<<blocks,threads,3*sizeof(double)*fwKernel*fwKernel>>>( d_image, d_kernel, dcrdata , d_cMask, d_mRdata, kcStep, hwKernel,xSize,ySize,nsteps_x, kerFracMask);

}

int ncompBG = (nCompKer - 1) * (((kerOrder + 1) * (kerOrder + 2)) / 2) + 1;

get_back<<<blocks_gpu,fwKernel*fwKernel,sizeof(double)*(fwKernel*fwKernel+4)>>>(dcrdata, d_kersol, kcStep, rPixX, rPixY, hwKernel, ncompBG, bgOrder);

for (j1 = gpupart - 1; j1 < nsteps_y; j1++) {
j0 = j1 * kcStep + hwKernel;

for (i1 = 0; i1 < nsteps_x; i1++) {
i0 = i1 * kcStep + hwKernel;

make_kernel(i0 + hwKernel, j0 + hwKernel, *kernel_sol);

#pragma omp parallel for private(j, i2,i, ni, qv, q, aks, uks, mbit, jc, jk, ic, ik, nc, kk )

for (j2 = 0; j2 < kcStep; j2++) {
	j = j0 + j2;
	if (j < ySize - hwKernel) {

		for (i2 = 0; i2 < kcStep; i2++) {
			i = i0 + i2;
			if (i < xSize - hwKernel) {

				ni = i + xSize * j;
				qv = q = aks = uks = 0.0;
				mbit = 0x0;
				for (jc = j - hwKernel; jc <= j + hwKernel; jc++) {
					jk = j - jc + hwKernel;

					for (ic = i - hwKernel; ic <= i + hwKernel; ic++) {
						ik = i - ic + hwKernel;

						nc = ic + xSize * jc;
						kk = kernel[ik + jk * fwKernel];
						q += image[nc] * kk;
						if (dovar) {
							if (convolveVariance)
								qv += (*variance)[nc] * kk;
							else
								qv += (*variance)[nc] * kk * kk;
						}
						mbit |= cMask[nc];
						aks += fabs(kk);
						if (!(cMask[nc] & FLAG_INPUT_ISBAD)) {
							uks += fabs(kk);
						}
					}
				}

				cRdata[ni] = q;
				if (dovar)
					vData[ni] = qv;

				/* mask propagation changed in 5.1.9 */
				/* mRData[ni]  |= mbit; */
				/* mRData[ni]  |= FLAG_OK_CONV      * (mbit > 0);*/
				mRData[ni] |= cMask[ni];
				mRData[ni] |= FLAG_OUTPUT_ISBAD
						* ((cMask[ni] & FLAG_INPUT_ISBAD) > 0);

				if (mbit) {
					if ((uks / aks) < kerFracMask) {
						mRData[ni] |= (FLAG_OUTPUT_ISBAD | FLAG_BAD_CONV);
					} else {
						mRData[ni] |= FLAG_OK_CONV;
					}
				}

			}
		}
	}
}
}
}

#pragma omp parallel for private(k)
for (l = gpupart * kcStep; l < rPixY - hwKernel; l++)
for (k = hwKernel; k < rPixX - hwKernel; k++)
cRdata[k + rPixX * l] += get_background(k, l, *kernel_sol);

CUDA_SAFE_CALL(
cudaMemcpy(cRdata, dcrdata, sizeof(float) * xSize * gpupart * kcStep,
cudaMemcpyDeviceToHost));
CUDA_SAFE_CALL(
cudaMemcpy(mRData, d_mRdata, sizeof(int) * xSize * gpupart * kcStep,
cudaMemcpyDeviceToHost));

cudaFree(dcrdata);
cudaFree(d_cMask);
cudaFree(d_mRdata);

if (dovar) {
CUDA_SAFE_CALL(
	cudaMemcpy(vData, d_vData, sizeof(float) * xSize * gpupart * kcStep,
			cudaMemcpyDeviceToHost));
cudaFree(d_vData);
cudaFree(d_variance);
free(*variance);
*variance = vData;
}

return;
}

extern "C" void cuda_finish(float *dtRData, float *diRData, double *iKerSol,
double *tKerSol) {
CUDA_SAFE_CALL (cudaFree(d_xstamp));CUDA_SAFE_CALL( cudaFree(d_ystamp));
CUDA_SAFE_CALL( cudaFree(dtRData));
CUDA_SAFE_CALL( cudaFree(diRData));
CUDA_SAFE_CALL( cudaFree(d_wxy));
CUDA_SAFE_CALL( cudaFree(d_matrix));
CUDA_SAFE_CALL( cudaFree(d_ksol));
CUDA_SAFE_CALL( cudaFree(d_sflag));
CUDA_SAFE_CALL( cudaFree(d_fx));
CUDA_SAFE_CALL( cudaFree(d_fy));
CUDA_SAFE_CALL( cudaFree(d_kercoe));
CUDA_SAFE_CALL( cudaFree(d_kernel));
CUDA_SAFE_CALL( cudaFree(dkernel_vec));

CUDA_SAFE_CALL( cudaFreeHost(fx));
CUDA_SAFE_CALL( cudaFreeHost(fy));
CUDA_SAFE_CALL( cudaFreeHost(mystamp));
CUDA_SAFE_CALL( cudaFreeHost(xstamp));
CUDA_SAFE_CALL( cudaFreeHost(ystamp));
CUDA_SAFE_CALL( cudaFreeHost(indx));

cudaDeviceReset();
return;
}
