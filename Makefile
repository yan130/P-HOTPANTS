#############################################################
# TO BE CHANGED BY EACH USER TO POINT TO include/ AND lib/ 
# DIRS HOLDING CFITSIO *.h AND libcfitsio IF THEY ARE NOT IN 
# THE STANDARD PLACES
# 

CFITSIOINCDIR =  ../include/
LIBDIR        =  /usr/lib/$(ARCH)
CUTIL_HOME      = /data/gbai/SDK/SDK/C
CUDA_HOME       = /usr/local/cuda-5.5
NVCCFLAGS       = -O2 -arch=sm_20


#
#
#############################################################
# COMPILATION OPTIONS BELOW
# 

# another good memory checker is valgrind : http://valgrind.kde.org/index.html
# valgrind --tool=memcheck hotpants

# for memory checking with libefence
# LIBS  = -L$(LIBDIR) -lm -lcfitsio -lefence

# for profiling with gprof
# COPTS = -pg -fprofile-arcs -funroll-loops -O3 -ansi -pedantic-errors -Wall -I$(CFITSIOINCDIR) 

# for gdbugging
#COPTS = -g3 -funroll-loops -O3 -ansi -pedantic-errors -Wall -I$(CFITSIOINCDIR) 

# standard usage
COPTS = -funroll-loops -O3 -std=c99 -pedantic-errors -Wall -I$(CFITSIOINCDIR)
LIBS  = -L../lib -L/usr/local/cuda/lib64 -L$(CUTIL_HOME)/lib -lcutil_x86_64 -lm -lcfitsio -lcudart  -lstdc++

INCLUDE =  -I./ -I$(CUDA_HOME)/include -I$(CUTIL_HOME)/common/inc/
# compiler
CC    = gcc 
NVCC  = nvcc
#
#
############################################################# 
# BELOW SHOULD BE OK, UNLESS YOU WANT TO COPY THE EXECUTABLES
# SOMEPLACE AFTER THEY ARE BUILT eg. hotpants
#

STDH  = functions.h globals.h defaults.h
ALL   = main.o vargs.o alard.o functions.o gpu_kernel.o 

all:	hotpants
# extractkern maskim

hotpants: $(ALL)
	$(CC) $(ALL) -fopenmp  -o hotpants $(LIBS) $(COPTS)
#	cp hotpants ../../bin/$(ARCH)

main.o: $(STDH) main.c
	$(CC) $(COPTS)  -fopenmp  -c main.c    

alard.o: $(STDH) alard.c
	$(CC) $(COPTS)  -fopenmp  -c alard.c    
#alard.o: $(STDH) alard.c
#	$(NVCC)  -O3  -c alard.c  -I$(CFITSIOINCDIR) 

functions.o: $(STDH) functions.c
	$(CC) $(COPTS)  -fopenmp  -c functions.c

vargs.o: $(STDH) vargs.c
	$(CC) $(COPTS)  -c vargs.c

gpu_kernel.o: gforg.h defaults.h gpu_kernel.cu
	$(NVCC)  -Xcompiler  -fopenmp  -G  -c gpu_kernel.cu  $(NVCCFLAGS) $(INCLUDE)

#gmatrix.o: functions.h gforg.h defaults.h gmatrix.cu
#	$(NVCC)  -c gmatrix.cu  -I/usr/wanmeng/NVIDIA_GPU_Computing_SDK/CUDALibraries/common/inc/

#extractkern : extractkern.o 
#	$(CC) extractkern.o -o extractkern $(LIBS) $(COPTS)

#extractkern.o : $(STDH) extractkern.c
#	$(CC) $(COPTS)  -c extractkern.c

#maskim : maskim.o
#	$(CC) maskim.o -o maskim $(LIBS) $(COPTS)

#maskim.o: $(STDH) maskim.c
#	$(CC) $(COPTS)  -c maskim.c

clean :
	rm -f *.o
	rm -f *~ .*~
	rm -f hotpants
	rm -f extractkern
	rm -f maskim
