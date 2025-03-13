# Makefile to build and execute Arm PL example programs
#
# Arm Performance Libraries version 24.10
# SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
# SPDX-FileCopyrightText: Copyright 2015-2024 NAG

# Arm recommends the use of -mcpu=native to give the best performance,
# and this is used in all of the Makefiles for Arm PL examples.
#
# It is possible to build the examples for a target different to the host
# machine by overriding Makefile options on the command line. Here we use the
# -mcpu flag to choose the target. At runtime, however, the target specific
# routines are selected based on the hardware of the host on which the
# examples are executed.
#
# Generic AArch64:
#     make FFLAGS="-mcpu=generic -I$(ARMPL_DIR)/include -fopenmp " \
           FLINKFLAGS="-fopenmp -mcpu=generic" CLINKFLAGS="-fopenmp -mcpu=generic" \
           CFLAGS="-fopenmp -mcpu=generic -I$(ARMPL_DIR)/include -std=c99"
# Neoverse N1:
#     make FFLAGS="-mcpu=neoverse-n1 -I$(ARMPL_DIR)/include -fopenmp " \
           FLINKFLAGS="-fopenmp -mcpu=neoverse-n1" CLINKFLAGS="-fopenmp -mcpu=neoverse-n1 " \
           CFLAGS="-fopenmp -mcpu=neoverse-n1 -I$(ARMPL_DIR)/include -std=c99"
# Neoverse V1:
#     make FFLAGS="-mcpu=neoverse-v1 -I$(ARMPL_DIR)/include -fopenmp " \
           FLINKFLAGS="-fopenmp -mcpu=neoverse-v1" CLINKFLAGS="-fopenmp -mcpu=neoverse-v1 " \
           CFLAGS="-fopenmp -mcpu=neoverse-v1 -I$(ARMPL_DIR)/include -std=c99"
#
# To make all examples run:
#     make all
#
# To make only C examples run:
#     make c_examples
#
# To make only Fortran examples run:
#     make f_examples
#
# For full documentation of the -armpl, -mcpu and -march flags see the compiler
# man pages.


ifeq ($(ARMPL_DIR),)
  $(error "ARMPL_DIR not set, try loading the appropriate module or explicitly exporting the variable")
endif

F90 := gfortran
FLINK ?= $(F90)
FFLAGS ?=  -fopenmp -mcpu=native -I$(ARMPL_DIR)/include
FLINKFLAGS ?= -fopenmp -mcpu=native
FLINKLIBS ?= -L$(ARMPL_DIR)/lib -larmpl_mp -lamath -lm
CC := gcc
CLINK ?= $(CC)
CFLAGS ?=  -fopenmp -mcpu=native -I$(ARMPL_DIR)/include -std=c99
CLINKFLAGS ?= -fopenmp -mcpu=native
CLINKLIBS ?= -L$(ARMPL_DIR)/lib -larmpl_mp -lamath -lm

EXPORT_PATH ?= LD_LIBRARY_PATH=$(ARMPL_DIR)/lib:$(LD_LIBRARY_PATH)
RUNNER ?=

obj := o

.SUFFIXES: .f90 .c .$(obj) .exe .res .diff .expected

# Source code of Fortran example programs
FSOURCE := armplinfo.f90 \
	sgetrf_example.f90 dgetrf_example.f90 domatcopy_f_example.f90 \
	sgemm_batch_example.f90 fftw_dft_r2c_1d_example.f90 \
	fftw_dft_1d_example.f90 sparse_spmv_f_example.f90 \
	sparse_spmm_f_example.f90 sparse_spadd_f_example.f90 \
	sparse_sprot_f_example.f90 \
	sparse_sptrsv_f_example.f90 \
	dtpmv_f_example.f90 dspmv_f_example.f90 \
	zhpmv_f_example.f90 dspr2_f_example.f90 \
	zhpr2_f_example.f90 dspr_f_example.f90 zspr_f_example.f90 \
	zhpr_f_example.f90 zimatcopy_f_example.f90 \
	workflow_f_example.f90

FEXE := $(FSOURCE:.f90=.exe)

# Source code of C example programs
CSOURCE := openrng.c armpl_dgemm_interleave_batch_c_example.c \
	dgesdd_c_example.c sgesdd_c_example.c dgetrf_c_example.c \
	sgetrf_c_example.c zgemm_batch_c_example.c \
	fftw_dft_r2c_1d_c_example.c fftw_dft_r2c_2d_c_example.c \
	fftw_dft_r2c_3d_c_example.c fftw_dft_1d_c_example.c \
	fftwf_dft_1d_c_example.c fftw_dft_2d_c_example.c \
	fftw_dft_3d_c_example.c fftw_many_dft_c_example.c \
	fftw_many_dft_r2c_c_example.c fftwf_dft_r2c_1d_c_example.c \
	fftw_guru_dft_r2c_c_example.c fftw_guru_dft_c2c_c_example.c \
	fftw_guru_split_dft_c_example.c fftw_guru_split_dft_r2c_c_example.c \
	sparse_spmv_c_example.c sparse_spmm_c_example.c \
	sparse_spadd_c_example.c sparse_sprot_c_example.c \
	sparse_sptrsv_c_example.c sparse_sptrsv_bad_c_example.c \
	dtbsv_c_example.c dtpmv_c_example.c dspmv_c_example.c \
	zhpmv_c_example.c dspr2_c_example.c zhpr2_c_example.c \
	dspr_c_example.c zdotu_c_example.c zspr_c_example.c \
	zhpr_c_example.c workflow_c_example.c

# Use of vector functions in libamath require binutils version >= 2.33 on Linux
# (https://github.com/bminor/binutils-gdb/commit/2301ed1), so check whether we
# can run the amath example before adding it to the list.
BINUTILS_VER=$(shell nm --version | head -1 |  sed -n -e 's/.*GNU.*\([0-9]\)\.\([0-9][0-9]\).*/\1\2/p')

# If the binutils version is not set then we're most likely on macOS
ifeq ($(BINUTILS_VER),)
    USE_AMATH=true
else
# Otherwise, check we have binutils >= 2.33
    $(info set - linux)
    USE_AMATH=$(shell [ $(BINUTILS_VER) -ge 233 ] && echo true)
endif

ifeq ($(USE_AMATH),true)
    CSOURCE += amath.c
endif

CEXE := $(CSOURCE:.c=.exe)

RESULTS := $(FSOURCE:.f90=.res) $(CSOURCE:.c=.res)
CRESULTS := $(CSOURCE:.c=.res)
FRESULTS := $(FSOURCE:.f90=.res)

all: diffs
c_examples: cdiffs
f_examples: fdiffs

sparse_export_ftn_libgfortran_ne_5.o: $(ARMPL_DIR)/src/sparse_export_ftn_libgfortran_ne_5.c
	@$(CC) -c $(CFLAGS) -o $@ $^

armpl_library.mod: armpl_blas.mod armpl_lapack.mod armpl_sparse.mod
armpl_sparse.mod: armpl_kinds.mod armpl_sparse_params.mod
armpl_sparse_params.mod: armpl_kinds.mod

FORTRAN_MOD_FLAGS :=  -fsyntax-only -J . 

%.mod: $(ARMPL_DIR)/include/%.f90
	$(F90) $(FORTRAN_MOD_FLAGS) $<

# Rule to compile/run a Fortran example program and produce a .res file
%.res: %.f90 sparse_export_ftn_libgfortran_ne_5.o armpl_library.mod
	@echo ""
	@echo "Compiling program $*.f90:"
	$(F90) -c $(FFLAGS) $*.f90 -o $*.$(obj)
	@echo "Linking program $*.exe:"
	@if [ "$*" = "sparse_spadd_f_example" ] || [ "$*" = "sparse_spmm_f_example" ] ; then\
	 echo "Linking in sparse export wrappers for gfortran, in case we have an old version of gcc/gfortran:" ;\
	 echo "$(FLINK) $(FLINKFLAGS) $*.$(obj) sparse_export_ftn_libgfortran_ne_5.o $(FLINKLIBS) -o $*.exe" ;\
	 $(FLINK) $(FLINKFLAGS) $*.$(obj) sparse_export_ftn_libgfortran_ne_5.o $(FLINKLIBS) -o $*.exe ;\
	else\
	 echo "$(FLINK) $(FLINKFLAGS) $*.$(obj) $(FLINKLIBS) -o $*.exe" ;\
	 $(FLINK) $(FLINKFLAGS) $*.$(obj) $(FLINKLIBS) -o $*.exe ;\
	fi
	@echo "Running program $*.exe:"
	$(EXPORT_PATH) $(RUNNER) ./$*.exe > $*.res
#	Show the results
	@cat $@
#	Compare the results with the expected results
	@diff -w $@ $*.expected | cat > $*.diff
#	Ignore the armplinfo example which will almost always be different
	@rm -f armplinfo.diff
#	Delete the diff file if it is empty
	@if [ -s $*.diff ]; then \
	  echo "!!! $* has diffs !!!"; \
	else \
	  rm -f $*.diff; \
	fi

# Rule to compile/run a C example program and produce a .res file
.c.res:
	@echo ""
	@echo "Compiling program $*.c:"
	$(CC) -c $(CFLAGS) $*.c -o $*.$(obj)
	@echo "Linking program $*.exe:"
	$(CLINK) $(CLINKFLAGS) $*.$(obj) $(CLINKLIBS) -o $*.exe
	@echo "Running program $*.exe:"
	@if [ "$*" = "amath" ]; then \
	  echo "For glibc versions older than 2.34, the amath example needs" ; \
	  echo "to be run with lazy binding disabled: LD_BIND_NOW=1." ; \
	  $(EXPORT_PATH) LD_BIND_NOW=1 $(RUNNER) ./$*.exe > $*.res ; \
	else \
	  $(EXPORT_PATH) $(RUNNER) ./$*.exe > $*.res ; \
	fi
#	Show the results
	@cat $@
#	Compare the results with the expected results
	@diff -w $@ $*.expected | cat > $*.diff
#	Delete the diff file if it is empty
	@if [ -s $*.diff ]; then \
	  echo "!!! $* has diffs !!!"; \
	else \
	  rm -f $*.diff; \
	fi

define test_results  =
@(ndiff=`ls *.diff 2> /dev/null | wc -w | sed -e 's/ //g'` ; \
  echo ""; \
  if [ "$${ndiff}" = "0" ]; then \
    echo "Testing: no example difference files were generated."; \
    echo "Test passed OK"; \
  else \
    echo "Testing: $${ndiff} example difference files were generated:"; \
    ls -l *.diff; \
    false; \
  fi)
endef

diffs: $(RESULTS)
	$(test_results)

cdiffs: $(CRESULTS)
	$(test_results)

fdiffs: $(FRESULTS)
	$(test_results)

clean:
	rm -f $(FEXE) $(CEXE) \
	      $(FSOURCE:.f90=.$(obj)) $(CSOURCE:.c=.$(obj)) \
	      $(RESULTS) *.diff *.mod
