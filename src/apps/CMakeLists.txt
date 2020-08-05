###############################################################################
# Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

blt_add_library(
  NAME apps
  SOURCES AppsData.cpp
          ENERGY.cpp
          ENERGY-Seq.cpp
          ENERGY-Cuda.cpp 
          ENERGY-OMP.cpp 
          ENERGY-OMPTarget.cpp 
          PRESSURE.cpp 
          PRESSURE-Seq.cpp 
          PRESSURE-Cuda.cpp 
          PRESSURE-OMP.cpp 
          PRESSURE-OMP.cpp 
  DEPENDS_ON common ${RAJA_PERFSUITE_DEPENDS}
  )
