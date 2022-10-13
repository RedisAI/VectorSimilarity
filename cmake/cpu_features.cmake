include(FetchContent)
option(BUILD_TESTING "" OFF)
option(CMAKE_POSITION_INDEPENDENT_CODE "" ON)
FetchContent_Declare(
	cpu_features
	GIT_REPOSITORY  https://github.com/google/cpu_features.git
	GIT_TAG  438a66e41807cd73e0c403966041b358f5eafc68
)
FetchContent_MakeAvailable(cpu_features)
