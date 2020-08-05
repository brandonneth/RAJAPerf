###############################################################################
# Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

blt_add_library(
  NAME lcals
  SOURCES 
          GEN_LIN_RECUR.cpp 
          GEN_LIN_RECUR-Seq.cpp 
          GEN_LIN_RECUR-Cuda.cpp
          GEN_LIN_RECUR-OMP.cpp
          GEN_LIN_RECUR-OMPTarget.cpp
          HYDRO_2D.cpp 
          HYDRO_2D-Seq.cpp 
          HYDRO_2D-Cuda.cpp
          HYDRO_2D-OMP.cpp
          HYDRO_2D-OMPTarget.cpp
  DEPENDS_ON common ${RAJA_PERFSUITE_DEPENDS}
  )
