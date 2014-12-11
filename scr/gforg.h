typedef struct
{
   int       x0,y0;       /* origin of stamp in region coords*/
   int       x,y;         /* center of stamp in region coords*/
   int       nx,ny;       /* size of stamp */
   int       *xss;        /* x location of test substamp centers */
   int       *yss;        /* y location of test substamp centers */
   int       nss;         /* number of detected substamps, 1 .. nss     */
   int       sscnt;       /* represents which nss to use,  0 .. nss - 1 */
   double    **vectors;   /* contains convolved image data */
   double    *krefArea;   /* contains kernel substamp data */
   double    **mat;       /* fitting matrices */
   double    *scprod;     /* kernel sum solution */
   double    sum;         /* sum of fabs, for sigma use in check_stamps */
   double    mean;
   double    median;
   double    mode;        /* sky estimate */
   double    sd;
   double    fwhm;
   double    lfwhm;
   double    chi2;        /* residual in kernel fitting */
   double    norm;        /* kernel sum */
   double    diff;        /* (norm - mean_ksum) * sqrt(sum) */
} stamp_struct;

/* GLOBAL VARS POSSIBLY SET ON COMMAND LINE */


int       hwKernel;
int       nKSStamps,hwKSStamp;
int       kerOrder, bgOrder;
int       convolveVariance;
float     kerFracMask, fillVal, fillValNoise;
char      *forceConvolve;

/* GLOBAL VARS NOT SET ON COMMAND LINE */
int       ngauss, *deg_fixe;
float     *sigma_gauss;
int       rPixX, rPixY;
int       nStamps, nCompKer, nC;
int       nComp, nCompBG, nBGVectors, nCompTotal;
int       pixStamp, fwKernel, fwStamp, hwStamp, fwKSStamp, kcStep, *indx;
int       cmpFile;
double    **wxy,*kernel_coeffs,*kernel;
char *figMerit;
/* REGION SIZED */
int       *mRData;   /* bad input data mask */

/* armin */
/* a dummy varialbe to do some testing */

/* verbose for debugging */
int        verbose;
float      kerSigReject;

//GPU
int       *dig,*didegx,*didegy,*dren;
int       *d_xstamp, *d_ystamp;
int       *mystamp,*xstamp, *ystamp;

extern double  *check_stack, **kernel_vec;
double   *dkernel_vec, * d_kercoe, *d_kernel, *dfilter_x,*dfilter_y;

double   * d_wxy;
double   * d_matrix;
double   * d_ksol;

int      * d_sflag;
double   *d_fx,*d_fy;
double   *fx,*fy;
float     rPixX2, rPixY2;
int       ncomp1,ncomp2, ncomp, nbg_vec;;
int       mat_size,vsize;
//float     *dsig;
