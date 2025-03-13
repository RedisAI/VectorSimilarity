/*
 * fftw_guru_split_dft: complex FFT using split guru interface
 *
 * Arm Performance Libraries version 24.10
 * SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates

 * This example demonstrates performing FFTs on a 3-d (2x3x2) space of 6x3 matrices.
 * A single matrix is created and then duplicated into the 3-d space.
 * Forwards and inverse FFTs are performed before the result is scaled and printed.
 * The computed values should match those in the original matrix.
 */

#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// These values must remain fixed in this example
// We are performing a 2-d transform (over a matrix)
#define RANK 2
// The matrix is repeated over a 3-d space
#define HM_RANK 3

// A helper function to duplicate the matrix into the 3-d domain
void duplicate_mat(double *orig, double *x, int n0, int n1, int hm2) {

	for (int i=0; i<n0; i++) {
		memcpy(&x[i*hm2*n1], &orig[i*n1], sizeof(double)*n1);
	}
}

// A helper function to print matrices in the 3-d domain following FFT execution
void print_mat(double *x, int n0, int n1, int hm2) {

	for (int i=0; i<n0; i++) {
		for (int k=0; k<hm2; k++) {
			for (int j=0; j<n1; j++) {
				printf(" %4.1f", x[i*hm2*n1 + k*n1 + j]);
			}
			if (k<hm2-1) printf("    ");
		}
		printf("\n");
	}
}

int main(void) {

	// Dimensions of the matrix
	const int n0 = 6;
	const int n1 = 3;

	// Dimensions of the 3-d space of matrices
	const int hm0 = 2;
	const int hm1 = 3;
	const int hm2 = 2;

	fftw_iodim dims[RANK];

	// Populate the dims array
	dims[0].n = n0;        dims[1].n = n1;
	dims[0].is = n1*hm2;   dims[1].is = 1;
	dims[0].os = n1*hm2;   dims[1].os = 1;

	fftw_iodim hm_dims[HM_RANK];

	// Populate the howmany_dims array
	hm_dims[0].n = hm0;              hm_dims[1].n = hm1;          hm_dims[2].n = hm2;
	hm_dims[0].is = n1*n0*hm2*hm1;   hm_dims[1].is = n1*n0*hm2;   hm_dims[2].is = n1;
	hm_dims[0].os = n1*n0*hm2*hm1;   hm_dims[1].os = n1*n0*hm2;   hm_dims[2].os = n1;

	// An input matrix, to be duplicated
	double *orig_r = fftw_alloc_real(n0*n1); // Real components
	double *orig_i = fftw_alloc_real(n0*n1); // Imaginary components
	for (int i=0; i<n0*n1; i++) {
		orig_r[i] = i + 1.0;
		orig_i[i] = i / 10.0 + 1.0;
	}

	// Duplicate the matrix over the multidimensional space of transform
	// domains of dimension hm0*hm1*hm2
	double *xr = fftw_alloc_real(n0*n1*hm0*hm1*hm2); // Real components
	double *xi = fftw_alloc_real(n0*n1*hm0*hm1*hm2); // Imaginary components
	for (int k=0; k<hm0; k++) {
		for (int j=0; j<hm1; j++) {
			for (int i=0; i<hm2; i++) {
				duplicate_mat(orig_r, &xr[k*hm2*hm1*n1*n0 + j*hm2*n1*n0 + i*n1], n0, n1, hm2);
				duplicate_mat(orig_i, &xi[k*hm2*hm1*n1*n0 + j*hm2*n1*n0 + i*n1], n0, n1, hm2);
			}
		}
	}

	printf("ARMPL example: FFT of a complex sequence using fftw_plan_guru_split_dft\n");
	printf("-----------------------------------------------------------------------\n\n");
	printf("Original matrix, duplicated over the 3-d space:\n\n");
	printf("Real component\n");
	print_mat(orig_r, n0, n1, 1);
	printf("\n");
	printf("Imaginary component\n");
	print_mat(orig_i, n0, n1, 1);
	printf("\n");
	fftw_free(orig_r);
	fftw_free(orig_i);

	// Create the arrays for output of the transform
	double *yr = fftw_alloc_real(n0*n1*hm0*hm1*hm2); // Real components
	double *yi = fftw_alloc_real(n0*n1*hm0*hm1*hm2); // Imaginary components

	// Create the FFTW plans
	fftw_plan forward = fftw_plan_guru_split_dft(RANK, dims, HM_RANK, hm_dims,
	                                             xr, xi, yr, yi, FFTW_ESTIMATE);

	// Backward plan takes output from forward plan and inverts the real and imaginary inputs/outputs
	fftw_plan backward = fftw_plan_guru_split_dft(RANK, dims, HM_RANK, hm_dims,
	                                              yi, yr, xi, xr, FFTW_ESTIMATE);

	// Execute the FFTs
	fftw_execute(forward);
	fftw_execute(backward);

	// Scale the output
	for (int i=0; i<n0*n1*hm0*hm1*hm2; i++) {
		xr[i] /= n0*n1;
		xi[i] /= n0*n1;
	}

	// Print each of the transformed matrices in layout order
	printf("Reconstructed, scaled matrices in the 3-d space:\n\n");
	printf("Real component.\n");
	for (int k=0; k<hm0; k++) {
		for (int j=0; j<hm1; j++) {
			for (int i=0; i<hm2; i++) {
				printf(" Matrix %2d:", k*hm1*hm2 + j*hm2 + i);
				if (i<hm2-1) printf("        ");
			}
			printf("\n");
			print_mat(&xr[k*hm2*hm1*n1*n0 + j*hm2*n1*n0], n0, n1, hm2);
			printf("\n");
		}
	}
	printf("\n");
	printf("Imaginary component.\n");
	for (int k=0; k<hm0; k++) {
		for (int j=0; j<hm1; j++) {
			for (int i=0; i<hm2; i++) {
				printf(" Matrix %2d:", k*hm1*hm2 + j*hm2 + i);
				if (i<hm2-1) printf("        ");
			}
			printf("\n");
			print_mat(&xi[k*hm2*hm1*n1*n0 + j*hm2*n1*n0], n0, n1, hm2);
			printf("\n");
		}
	}
	printf("\n");

	// Cleanup
	fftw_destroy_plan(forward);
	fftw_destroy_plan(backward);

	fftw_free(xr);
	fftw_free(xi);
	fftw_free(yr);
	fftw_free(yi);

	return 0;
}
