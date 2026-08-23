# Probe whether the current toolchain can actually build a SIMD tier, instead of inferring the
# toolchain's capability from a side channel (a compiler flag check alone, or a binutils version
# table). CHECK_CXX_COMPILER_FLAG only asks the compiler whether it recognizes a flag; it does
# not ask whether the compiler and assembler can carry a real translation unit through to a
# finished object under the tier's complete flag combination. This probe does that: it
# try_compiles the tier's own source file under the tier's own flags, so a flag combination that
# the compiler accepts individually but rejects together, or a flag whose instructions the
# assembler cannot emit, fails here rather than reaching the build.
#
#   vecsim_tier_compiles(<result_var> SOURCE <path> FLAGS <flag string> [NAME <label>])
#
# Sets <result_var> to true/false depending on whether <path> compiles under <flag string>.
# Callers should define each tier's flag string once in a variable and pass that same variable
# both here and to set_source_files_properties(), so the probe can never test a different flag
# set than the one the build actually uses.

include(CheckCXXCompilerFlag)

function(vecsim_tier_compiles result_var)
	set(_one_value_args SOURCE FLAGS NAME)
	cmake_parse_arguments(_tier_probe "" "${_one_value_args}" "" ${ARGN})

	if(NOT _tier_probe_SOURCE)
		message(FATAL_ERROR "vecsim_tier_compiles: SOURCE is required")
	endif()
	if(NOT _tier_probe_NAME)
		set(_tier_probe_NAME "${_tier_probe_SOURCE}")
	endif()

	# The cache variable name follows the same idea CHECK_CXX_COMPILER_FLAG relies on (cache the
	# answer under a name that identifies what was asked), extended here with the compiler
	# identity/version and the flag string themselves, so a compiler upgrade or an edit to the
	# tier's flags cannot reuse a stale answer computed under a different compiler or different
	# flags.
	string(MAKE_C_IDENTIFIER
		"TIER_COMPILES_${_tier_probe_NAME}_${CMAKE_CXX_COMPILER_ID}_${CMAKE_CXX_COMPILER_VERSION}_${_tier_probe_FLAGS}"
		_tier_probe_cache_var)

	if(NOT DEFINED ${_tier_probe_cache_var})
		# Resolve relative to the caller's source directory (functions/*.cpp is written relative
		# to src/VecSim/spaces/CMakeLists.txt), since try_compile does not carry that context.
		get_filename_component(_tier_probe_source_abs "${_tier_probe_SOURCE}" ABSOLUTE)

		# cpu_features arrives via FetchContent_MakeAvailable(cpu_features) in
		# cmake/cpu_features.cmake, included before any tier block runs, so the target exists
		# here. Read its include directory from the target itself rather than hardcoding a path
		# under the FetchContent _deps tree, which is an implementation detail that can move.
		get_target_property(_tier_probe_cpu_features_dir cpu_features SOURCE_DIR)

		# The tier translation units have no main(): they are dispatch kernels selected at
		# runtime, not entry points. A default try_compile probe builds an executable and would
		# fail to LINK for every tier for that reason alone, which would silently disable the
		# whole dispatch layer while still reporting a successful configure. Building a static
		# library instead only requires the tier's own translation unit to compile and assemble.
		set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

		try_compile(${_tier_probe_cache_var}
			"${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/tier_probe/${_tier_probe_cache_var}"
			SOURCES "${_tier_probe_source_abs}"
			CMAKE_FLAGS
				"-DINCLUDE_DIRECTORIES:STRING=${root}/src;${_tier_probe_cpu_features_dir}/include"
			COMPILE_DEFINITIONS "${_tier_probe_FLAGS}"
			CXX_STANDARD ${CMAKE_CXX_STANDARD}
			CXX_STANDARD_REQUIRED ON
			OUTPUT_VARIABLE _tier_probe_output
		)

		if(NOT ${_tier_probe_cache_var})
			message(STATUS "Skipping tier ${_tier_probe_NAME}: toolchain failed to compile "
				"${_tier_probe_SOURCE} with '${_tier_probe_FLAGS}'. A tier that cannot compile "
				"on this toolchain is expected on some machines; the runtime degrades to a "
				"lower tier.")
		endif()
	endif()

	set(${result_var} ${${_tier_probe_cache_var}} PARENT_SCOPE)
endfunction()
