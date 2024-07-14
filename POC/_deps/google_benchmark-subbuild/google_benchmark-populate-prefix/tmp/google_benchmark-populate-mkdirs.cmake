# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/guy/Code/VectorSimilarity/POC/_deps/google_benchmark-src"
  "/home/guy/Code/VectorSimilarity/POC/_deps/google_benchmark-build"
  "/home/guy/Code/VectorSimilarity/POC/_deps/google_benchmark-subbuild/google_benchmark-populate-prefix"
  "/home/guy/Code/VectorSimilarity/POC/_deps/google_benchmark-subbuild/google_benchmark-populate-prefix/tmp"
  "/home/guy/Code/VectorSimilarity/POC/_deps/google_benchmark-subbuild/google_benchmark-populate-prefix/src/google_benchmark-populate-stamp"
  "/home/guy/Code/VectorSimilarity/POC/_deps/google_benchmark-subbuild/google_benchmark-populate-prefix/src"
  "/home/guy/Code/VectorSimilarity/POC/_deps/google_benchmark-subbuild/google_benchmark-populate-prefix/src/google_benchmark-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/guy/Code/VectorSimilarity/POC/_deps/google_benchmark-subbuild/google_benchmark-populate-prefix/src/google_benchmark-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/guy/Code/VectorSimilarity/POC/_deps/google_benchmark-subbuild/google_benchmark-populate-prefix/src/google_benchmark-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
