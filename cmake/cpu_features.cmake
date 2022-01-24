include(FetchContent)
option(BUILD_TESTING "" OFF)
option(CMAKE_POSITION_INDEPENDENT_CODE "" ON)
FetchContent_Declare(
	cpu_features
	GIT_REPOSITORY  https://github.com/google/cpu_features.git
	GIT_TAG f5a7bf80ae4c7df3285e23341ec5f37a5254095
)
FetchContent_MakeAvailable(cpu_features)
