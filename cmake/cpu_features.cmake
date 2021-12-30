project(cpu_features)

include(FetchContent)
FetchContent_Declare(
	cpu_features
	URL https://github.com/google/cpu_features/archive/refs/tags/v0.6.0.zip
)
FetchContent_MakeAvailable(cpu_features)
