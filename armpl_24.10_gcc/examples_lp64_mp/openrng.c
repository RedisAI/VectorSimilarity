/*
 * openrng - Random Number Generation
 *
 * Arm Performance Libraries version 24.10
 * SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
 */
#include <openrng.h>

#include <stdio.h>
#include <stdlib.h>

void assert_message(int condition, const char *message) {
	if (!condition) {
		printf("Error: %s\n", message);
		exit(EXIT_FAILURE);
	}
}

int main() {

	const size_t nIterations = 1000 * 1000;
	const size_t nRandomNumbers = 2 * nIterations;

	//
	// Declare and initialise the stream.
	//
	// In this example, we've selected the PHILOX4X32X10 generator and seeded it
	// with 42. We can then check that the method executed succesfully by checking
	// the return value for VSL_ERROR_OK. Most methods return VSL_ERROR_OK on
	// success.
	//
	VSLStreamStatePtr stream;
	int errcode = vslNewStream(&stream, VSL_BRNG_PHILOX4X32X10, 42);
	assert_message(errcode == VSL_ERROR_OK, "vslNewStream failed");

	//
	// Allocate a buffer for storing random numbers.
	//
	float *randomNumbers = (float *) malloc(nRandomNumbers * sizeof(float));
	assert_message(randomNumbers != NULL, "malloc failed");

	//
	// Generate a uniform distribution between 0 and 1.
	//
	// First, we select the method used to generate the uniform distribution; in
	// this example, we use the standard method. We pass in a pointer to an
	// initialised stream, the amount of random numbers we want, followed by a
	// pointer to a buffer big enough to hold all the random numbers requested.
	// Finally, we pass in parameters specific to the distribution, in this case,
	// 0 and 1, meaning we want the range [0, 1).
	//
	errcode = vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, nRandomNumbers, randomNumbers, 0, 1);
	assert_message(errcode == VSL_ERROR_OK, "vsRngUniform failed");

	//
	// Use the random numbers.
	//
	// This is a classic algorithm used for estimating the value of pi. We imagine
	// a unit square overlapping a quarter of a circle with unit radius. We then
	// treat pairs of successive random numbers as points on the unit square. We
	// can check if the point is inside the quarter circle by measuring the
	// distance between the point and the centre of the circle; if the distance is
	// less than 1, the point is inside the circle. The proportion of points
	// inside the circle should be
	//
	//  (area of quarter circle) / (area of square) := pi / 4.
	//
	// so
	//
	//  pi = 4 * (proportion of points inside circle)
	//
	int count = 0;
	for (size_t i = 0; i < nIterations; i++) {
		float x = randomNumbers[2 * i + 0];
		float y = randomNumbers[2 * i + 1];

		if (x * x + y * y < 1) {
			count++;
		}
	}
	float estimateOfPi = 4.0f * count / nIterations;

	printf("Estimate of pi:        %f\n", estimateOfPi);
	printf("Number of iterations:  %zu\n", nIterations);

	//
	// The buffer passed into vsRngUniform is still owned by the user.
	//
	free(randomNumbers);

	//
	// Release any resources held by the stream.
	//
	errcode = vslDeleteStream(&stream);
	assert_message(errcode == VSL_ERROR_OK, "vslDeleteStream failed");

	return EXIT_SUCCESS;
}