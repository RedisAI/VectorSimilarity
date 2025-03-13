/*
 * Arm Performance Libraries version 24.10
 * SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
 */

#ifndef ARMPL_STATUS_H
#define ARMPL_STATUS_H

#ifdef __cplusplus
extern "C" {
#endif

/* ENUMs */

typedef enum armpl_status {
	ARMPL_STATUS_SUCCESS=0,
	ARMPL_STATUS_INPUT_PARAMETER_ERROR=1,
	ARMPL_STATUS_EXECUTION_FAILURE=2,
} armpl_status_t;

#ifdef __cplusplus
} //extern "C"
#endif

#endif //ARMPL_STATUS_H
