/*
 * fftw_guru_dft_c2c: complex FFT using guru interface
 *
 * Arm Performance Libraries version 24.10
 * SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates

 * This example demonstrates performing FFTs on a 3-d (2x4x3) space of 3x4 matrices.
 * A single matrix is created and then duplicated into the 3-d space. A
 * complex-to-complex FFT is performed, followed by another complex-to-complex
 * FFT. The result is then scaled before printing so that the values should
 * match those in the original matrix.
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
void duplicate_mat(fftw_complex *orig, fftw_complex *x, int n0, int n1, int hm2) {

	for (int i=0; i<n0; i++) {
		memcpy(&x[i*hm2*n1], &orig[i*n1], sizeof(fftw_complex)*n1);
	}
}

// A helper function to print matrices in the 3-d domain following FFT execution
void print_mat(fftw_complex *x, int n0, int n1, int hm2) {

	for (int i=0; i<n0; i++) {
		for (int k=0; k<hm2; k++) {
			for (int j=0; j<n1; j++) {
				double real = x[i*hm2*n1 + k*n1 + j][0];
				double imag = x[i*hm2*n1 + k*n1 + j][1];
				printf("  % 5.1f +% 5.1fi", real, imag);
			}
			if (k<hm2-1) printf("     ");
		}
		printf("\n");
	}
}

int main(void) {

	// Dimensions of the matrix
	const int n0 = 3;
	const int n1 = 4;

	// Dimensions of the 3-d space of matrices
	const int hm0 = 2;
	const int hm1 = 4;
	const int hm2 = 3;

	fftw_iodim dims[RANK];

	// Populate the dims array
	dims[0].n = n0;        dims[1].n = n1;
	dims[0].is = n1*hm2;   dims[1].is = 1;
	dims[0].os = n1*hm2; dims[1].os = 1;

	fftw_iodim hm_dims[HM_RANK];

	// Populate the howmany_dims array
	hm_dims[0].n = hm0;              hm_dims[1].n = hm1;          hm_dims[2].n = hm2;
	hm_dims[0].is = n1*n0*hm2*hm1;   hm_dims[1].is = n1*n0*hm2;   hm_dims[2].is = n1;
	hm_dims[0].os = n1*n0*hm2*hm1; hm_dims[1].os = n1*n0*hm2; hm_dims[2].os = n1;

	// An input matrix, to be duplicated
	fftw_complex *orig = fftw_alloc_complex(n0*n1);
	for (int i=0; i<n0*n1; i++) {
		orig[i][0] = i + 1.0;
		orig[i][1] = i + 1.0;
	}

	// Duplicate the matrix over the multidimensional space of transform
	// domains of dimension hm0*hm1*hm2
	fftw_complex *x = fftw_alloc_complex(n0*n1*hm0*hm1*hm2);
	fftw_complex *y = fftw_alloc_complex(n0*n1*hm0*hm1*hm2);
	for (int k=0; k<hm0; k++) {
		for (int j=0; j<hm1; j++) {
			for (int i=0; i<hm2; i++) {
				duplicate_mat(orig, &x[k*hm2*hm1*n1*n0 + j*hm2*n1*n0 + i*n1], n0, n1, hm2);
			}
		}
	}

	printf("ARMPL example: FFT of a complex sequence using fftw_plan_guru_dft\n");
	printf("-----------------------------------------------------------------\n\n");
	printf("Original matrix, duplicated over the 3-d space:\n\n");
	print_mat(orig, n0, n1, 1);
	printf("\n");
	fftw_free(orig);

	// Create the FFTW plans
	fftw_plan forward  = fftw_plan_guru_dft(RANK, dims, HM_RANK, hm_dims, x, y, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_plan backward = fftw_plan_guru_dft(RANK, dims, HM_RANK, hm_dims, y, x, FFTW_BACKWARD, FFTW_ESTIMATE);

	// Execute the FFTs
	fftw_execute(forward);
	fftw_execute(backward);

	// Scale the output
	for (int i=0; i<n0*n1*hm0*hm1*hm2; i++) {
		x[i][0] /= n0*n1;
		x[i][1] /= n0*n1;
	}

	// Print each of the transformed matrices in layout order
	printf("Reconstructed, scaled matrices in the 3-d space:\n\n");
	for (int k=0; k<hm0; k++) {
		for (int j=0; j<hm1; j++) {
			for (int i=0; i<hm2; i++) {
				printf("    Matrix %2d:", k*hm1*hm2 + j*hm2 + i);
				if (i<hm2-1) printf("                                                   ");
			}
			printf("\n");
			print_mat(&x[k*hm2*hm1*n1*n0 + j*hm2*n1*n0], n0, n1, hm2);
			printf("\n");
		}
	}

	// Cleanup
	fftw_destroy_plan(forward);
	fftw_destroy_plan(backward);

	fftw_free(x);
	fftw_free(y);

	return 0;
}
