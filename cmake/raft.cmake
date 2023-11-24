if(USE_CUDA)
	# Set which version of RAPIDS to use
	set(RAPIDS_VERSION 23.12)
	# Set which version of RAFT to use (defined separately for testing
	# minimal dependency changes if necessary)
	set(RAFT_VERSION "${RAPIDS_VERSION}")
	set(RAFT_FORK "rapidsai")
	set(RAFT_PINNED_TAG "branch-${RAPIDS_VERSION}")

	# Download CMake file for bootstrapping RAPIDS-CMake, a utility that
	# simplifies handling of complex RAPIDS dependencies
	if (NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/RAPIDS.cmake)
		file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-${RAPIDS_VERSION}/RAPIDS.cmake
			${CMAKE_CURRENT_BINARY_DIR}/RAPIDS.cmake)
	endif()
	include(${CMAKE_CURRENT_BINARY_DIR}/RAPIDS.cmake)

	# General tool for orchestrating RAPIDS dependencies
	include(rapids-cmake)
	# CPM helper functions with dependency tracking
	include(rapids-cpm)
	rapids_cpm_init()
	# Common CMake CUDA logic
	include(rapids-cuda)
	# Include required dependencies in Project-Config.cmake modules
	# include(rapids-export)  TODO(wphicks)
	# Functions to find system dependencies with dependency tracking
	include(rapids-find)

	# Correctly handle supported CUDA architectures
	#    (From rapids-cuda)
	rapids_cuda_init_architectures(VectorSimilarity)

	# Find system CUDA toolkit
	rapids_find_package(CUDAToolkit REQUIRED)

	set(RAFT_VERSION "${RAPIDS_VERSION}")
	set(RAFT_FORK "rapidsai")
	set(RAFT_PINNED_TAG "branch-${RAPIDS_VERSION}")
	
	function(find_and_configure_raft)
		set(oneValueArgs VERSION FORK PINNED_TAG COMPILE_LIBRARY)
		cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
			"${multiValueArgs}" ${ARGN} )
	
		set(RAFT_COMPONENTS "")
		if(PKG_COMPILE_LIBRARY)
			string(APPEND RAFT_COMPONENTS " compiled")
		endif()
		# Invoke CPM find_package()
		#     (From rapids-cpm)
		rapids_cpm_find(raft ${PKG_VERSION}
			GLOBAL_TARGETS      raft::raft
			BUILD_EXPORT_SET    VectorSimilarity-exports
			INSTALL_EXPORT_SET  VectorSimilarity-exports
			COMPONENTS          ${RAFT_COMPONENTS}
			CPM_ARGS
			GIT_REPOSITORY https://github.com/${PKG_FORK}/raft.git
			GIT_TAG        ${PKG_PINNED_TAG}
			SOURCE_SUBDIR  cpp
			OPTIONS
			"BUILD_TESTS OFF"
			"BUILD_BENCH OFF"
			"RAFT_COMPILE_LIBRARY ${PKG_COMPILE_LIBRARY}"
		)
		if(raft_ADDED)
			message(VERBOSE "VectorSimilarity: Using RAFT located in ${raft_SOURCE_DIR}")
		else()
			message(VERBOSE "VectorSimilarity: Using RAFT located in ${raft_DIR}")
		endif()
	endfunction()
	
	# Change pinned tag here to test a commit in CI
	# To use a different RAFT locally, set the CMake variable
	# CPM_raft_SOURCE=/path/to/local/raft
	find_and_configure_raft(VERSION    ${RAFT_VERSION}.00
		FORK             ${RAFT_FORK}
		PINNED_TAG       ${RAFT_PINNED_TAG}
		COMPILE_LIBRARY  OFF
	)
endif()
