P-HOTPANTS
==========

we accelerate the online-available image subtraction package "HOTPANTS" on one GPU and one CPU, and try our best to achieve the best performance. 


we accelerate the online-available image subtraction package "HOTPANTS" (http://www.astro.washington.edu/users/becker/v2.0/c_software.html) on one GPU and one CPU, and try our best to achieve the best performance. 

In P-HOTPANTS, we use GPU to conduct most computation and CPU to check or make decisions. In the most time-consuming part, Convolving, we use both CPU and GPU and achieve an overall 4 times speedup on a desktop with an Intel i7 CPU and an NVIDIA GTX580 GPU.

We focus on image subtraction procedure and remove extractkern.c, extractkernOnes.c and maskim.c.


Compile
Both HOTPANTS and P-HOTPANTS were developed on Linux. Other operating systems may
require minor changes to the source code. Synthetic image is not recommended for inputs.
Modify the CFITIOS and CUTIL direction in MAKEFILE before conduct “make”.
Modify the GPU-PART parameter in gpu_kernel.cu to set the ratio of GPU workload in Convolving.


Sample
we provide one 1K x 1K and one 3K x 3K image are input, please use the following commands:
./hotpants -inim input_3K.fit -tmplim templ_3K.fit -outim resd_3K.fit -v 0
or 
./hotpants -inim input_1K.fit -tmplim templ_1K.fit -outim resd_1K.fit -v 0

Citation
Yan Zhao, Qiong Luo, Senhong Wang, Chao Wu, 
Accelerating Astronomical Image Subtraction on Heterogeneous Processors

Contact
P-HOTPANTS  was mainly written by Yan Zhao.
For bug-fixes, feedback please contact:
zhaoyan1555@gmail.com
 

