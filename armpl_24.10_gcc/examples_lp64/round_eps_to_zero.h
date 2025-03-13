#ifndef ROUND_EPS_TO_ZERO_H
#define ROUND_EPS_TO_ZERO_H

/*
 * Arm Performance Libraries version 24.10
 * SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
 */

#include <float.h>
#include <math.h>

// On some architectures we occasionally see tiny differences between an expected
// answer of 0 and the calculated answer, so we choose not to display these
float round_eps_to_zero_f(float x) {
	return fabs(x) > FLT_EPSILON ? x : 0;
}
double round_eps_to_zero_d(double x) {
	return fabs(x) > DBL_EPSILON ? x : 0;
}

#endif
