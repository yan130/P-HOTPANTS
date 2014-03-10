#include<stdio.h>
#include<string.h>
#include<math.h>
#include<malloc.h>
#include<stdlib.h>
#include<omp.h>

#include "defaults.h"
#include "globals.h"
#include "functions.h"

/*
  
  Several of these subroutines appear originally in code created by
  Cristophe Alard for ISIS, but have been modified and/or rewritten
  for the current software package.  In particular, the construction
  of the least squares matrices have been taken directly from the ISIS
  code.

  08/20/01 acbecker@physics.bell-labs.com
  
*/





void getFinalStampSig(stamp_struct *stamp, float *imDiff, float *imNoise, double *sig) {
   int i, j, idx, nsig=0;
   int xRegion2, xRegion = stamp->xss[stamp->sscnt];
   int yRegion2, yRegion = stamp->yss[stamp->sscnt];
   float idat, indat;

   *sig = 0;

   for (j = 0; j < fwKSStamp; j++) {
      yRegion2 = yRegion - hwKSStamp + j;
      
      for (i = 0; i < fwKSStamp; i++) {
	 xRegion2 = xRegion - hwKSStamp + i;

	 idx   = xRegion2+rPixX*yRegion2;
	 idat  = imDiff[idx];
	 indat = 1. / imNoise[idx];

	 /* this shouldn't be the case, but just in case... */
	 if (mRData[idx] & FLAG_INPUT_ISBAD)
	    continue;

	 nsig++;
	 *sig += idat * idat * indat * indat;

      }
   }
   if (nsig > 0) 
      *sig /= nsig;
   else
      *sig = -1;

   return;
}


extern  double make_kernel(int xi, int yi, double *kernelSol) {
   /*****************************************************
    * Create the appropriate kernel at xi, yi, return sum
    *****************************************************/
  //printf("%f--%f\n",kernel_vec[9][180],kernel_vec[48][200]); 
   int    i1,k,ix,iy,i;
   double ax,ay,sum_kernel;
   double xf, yf;
   
   k  = 2;
   /* RANGE FROM -1 to 1 */
   xf = (xi - 0.5 * rPixX) / (0.5 * rPixX);
   yf = (yi - 0.5 * rPixY) / (0.5 * rPixY);
  #pragma omp parallel for private(ix, iy, ax, ay)  
   for (i1 = 1; i1 < nCompKer; i1++) {
      kernel_coeffs[i1] = 0.0;
      ax = 1.0;
      for (ix = 0; ix <= kerOrder; ix++) {
         ay = 1.0;
         for (iy = 0; iy <= kerOrder - ix; iy++) {
            kernel_coeffs[i1] += kernelSol[k++] * ax * ay;
            ay *= yf;
         }
         ax *= xf;
      }
   }
   kernel_coeffs[0] = kernelSol[1]; 
   
   for (i = 0; i < fwKernel * fwKernel; i++)
      kernel[i] = 0.0;
   
   sum_kernel = 0.0;
  #pragma omp parallel for private(i1) reduction( +:sum_kernel)
   for (i = 0; i < fwKernel * fwKernel; i++) {
      for (i1 = 0; i1 < nCompKer; i1++) {
         kernel[i] += kernel_coeffs[i1] * kernel_vec[i1][i];
      }
      sum_kernel += kernel[i];    
   }
   return sum_kernel;
}

double get_background(int xi, int yi, double *kernelSol) {
   /*****************************************************
    * Return background value at xi, yi
    *****************************************************/
   
   double  background,ax,ay,xf,yf;
   int     i,j,k;
   int     ncompBG;

   ncompBG = (nCompKer - 1) * ( ((kerOrder + 1) * (kerOrder + 2)) / 2 ) + 1;
   
   background = 0.0;
   k          = 1;
   /* RANGE FROM -1 to 1 */
   xf = (xi - 0.5 * rPixX) / (0.5 * rPixX);
   yf = (yi - 0.5 * rPixY) / (0.5 * rPixY);
   
   ax=1.0;
   for (i = 0; i <= bgOrder; i++) {
      ay = 1.0; 
      for (j = 0; j <= bgOrder - i; j++) {
         background += kernelSol[ncompBG+k++] * ax * ay;
         /* fprintf(stderr, "bg: %d %d %d %d %f %f %f\n", xi, yi, i, j, ax, ay, kernelSol[ncompBG+k-1]); */
         ay *= yf;
      }
      ax *= xf;
   }
   return background;
}




extern  int ludcmp_d1(double *a, int n, int nn, int *indx)
#define TINY 1.0e-20;
{
int     i,imax=0,j,k;
   double  big,dum,sum,temp2;
   double  *vv,*lvector();
   void    lnrerror();
   vv=(double *)malloc((n+1)*sizeof(double));
   
  
   for (i=1;i<=n;i++) {
      big=0.0;
      for (j=1;j<=n;j++)
         if ((temp2=fabs(a[i*nn+j])) > big) big=temp2;
      if (big == 0.) {
         fprintf(stderr," Numerical Recipies run error....");
         fprintf(stderr,"Singular matrix in routine LUDCMP\n");
         fprintf(stderr,"Goodbye ! \n");

         return (1);
      }
      vv[i]=1.0/big;
   }
   for (j=1;j<=n;j++) {
      for (i=1;i<j;i++) {
         sum=a[i*nn+j];
         for (k=1;k<i;k++) sum -= a[i*nn+k]*a[k*nn+j];
         a[i*nn+j]=sum;
      }
      big=0.0;
      for (i=j;i<=n;i++) {
         sum=a[i*nn+j];
         for (k=1;k<j;k++)
            sum -= a[i*nn+k]*a[k*nn+j];
         a[i*nn+j]=sum;
         if ( (dum=vv[i]*fabs(sum)) >= big) {
            big=dum;
            imax=i;
         }
      }
      if (j != imax) {
for (k=1;k<=n;k++) {
            dum=a[imax*nn+k];
            a[imax*nn+k]=a[j*nn+k];
            a[j*nn+k]=dum;
         }
        
         vv[imax]=vv[j];
      }
      indx[j]=imax;
      if (a[j*nn+j] == 0.0) a[j*nn+j]=TINY;
      if (j != n) {
         dum=1.0/(a[j*nn+j]);
         for (i=j+1;i<=n;i++) a[i*nn+j] *= dum;
      }
   }
   free(vv);
   return 0;
}



extern  void lubksb_d1(double *a, int n, int nn, int *indx, double  *b)
{
   int i,ii=0,ip,j;
   double  sum;
   for (i=1;i<=n;i++) {
      ip=indx[i];
      sum=b[ip];
      b[ip]=b[i];
      if (ii)
         for (j=ii;j<=i-1;j++) sum -= a[i*nn+j]*b[j];
      else if (sum) ii=i;
      b[i]=sum;
   }
   for (i=n;i>=1;i--) {
      sum=b[i];
      for (j=i+1;j<=n;j++) sum -= a[i*nn+j]*b[j];
      b[i]=sum/a[i*nn+i];
   }
}
