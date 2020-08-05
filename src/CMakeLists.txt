###############################################################################
# Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

include_directories(.)

add_subdirectory(common)
add_subdirectory(apps)
add_subdirectory(lcals)
add_subdirectory(polybench)

set(RAJA_PERFSUITE_EXECUTABLE_DEPENDS
    common
    apps
    lcals
    polybench)
list(APPEND RAJA_PERFSUITE_EXECUTABLE_DEPENDS ${RAJA_PERFSUITE_DEPENDS}) 

if(ENABLE_TARGET_OPENMP)
  remove_definitions(-DRUN_RAJA_SEQ -DRUN_OPENMP )

blt_add_executable(
  NAME raja-perf-omptarget.exe
  SOURCES RAJAPerfSuiteDriver.cpp
  apps/AppsData.cpp
  apps/ENERGY.cpp
  apps/ENERGY-Seq.cpp
  apps/ENERGY-OMPTarget.cpp
  apps/PRESSURE.cpp
  apps/PRESSURE-Seq.cpp
  apps/PRESSURE-OMPTarget.cpp
  lcals/GEN_LIN_RECUR.cpp
  lcals/GEN_LIN_RECUR-Seq.cpp
  lcals/GEN_LIN_RECUR-OMPTarget.cpp
  lcals/HYDRO_2D.cpp
  lcals/HYDRO_2D-Seq.cpp
  lcals/HYDRO_2D-OMPTarget.cpp
  polybench/POLYBENCH_3MM.cpp
  polybench/POLYBENCH_3MM-Seq.cpp
  polybench/POLYBENCH_3MM-OMPTarget.cpp
  polybench/POLYBENCH_ATAX.cpp
  polybench/POLYBENCH_ATAX-Seq.cpp
  polybench/POLYBENCH_ATAX-OMPTarget.cpp
  polybench/POLYBENCH_HEAT_3D.cpp
  polybench/POLYBENCH_HEAT_3D-Seq.cpp
  polybench/POLYBENCH_HEAT_3D-OMPTarget.cpp
  polybench/POLYBENCH_JACOBI_1D.cpp
  polybench/POLYBENCH_JACOBI_1D-Seq.cpp
  polybench/POLYBENCH_JACOBI_1D-OMPTarget.cpp
  polybench/POLYBENCH_JACOBI_2D.cpp
  polybench/POLYBENCH_JACOBI_2D-Seq.cpp
  polybench/POLYBENCH_JACOBI_2D-OMPTarget.cpp
  polybench/POLYBENCH_MVT.cpp
  polybench/POLYBENCH_MVT-Seq.cpp
  polybench/POLYBENCH_MVT-OMPTarget.cpp
  common/DataUtils.cpp
  common/Executor.cpp
  common/KernelBase.cpp
  common/OutputUtils.cpp
  common/RAJAPerfSuite.cpp
  common/RPTypes.hpp
  common/RunParams.cpp
  DEPENDS_ON ${RAJA_PERFSUITE_DEPENDS}
)

else()
blt_add_executable(
  NAME raja-perf.exe
  SOURCES RAJAPerfSuiteDriver.cpp
  DEPENDS_ON ${RAJA_PERFSUITE_EXECUTABLE_DEPENDS}
  )
endif()
