###############################################################################
# Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

blt_add_library(
  NAME polybench
  SOURCES 
          POLYBENCH_3MM.cpp
          POLYBENCH_3MM-Seq.cpp
          POLYBENCH_3MM-Cuda.cpp
          POLYBENCH_3MM-OMP.cpp
          POLYBENCH_3MM-OMPTarget.cpp
          POLYBENCH_ATAX.cpp
          POLYBENCH_ATAX-Seq.cpp
          POLYBENCH_ATAX-Cuda.cpp
          POLYBENCH_ATAX-OMP.cpp
          POLYBENCH_ATAX-OMPTarget.cpp
          POLYBENCH_HEAT_3D.cpp
          POLYBENCH_HEAT_3D-Seq.cpp
          POLYBENCH_HEAT_3D-Cuda.cpp
          POLYBENCH_HEAT_3D-OMP.cpp
          POLYBENCH_HEAT_3D-OMPTarget.cpp
          POLYBENCH_JACOBI_1D.cpp
          POLYBENCH_JACOBI_1D-Seq.cpp
          POLYBENCH_JACOBI_1D-Cuda.cpp
          POLYBENCH_JACOBI_1D-OMP.cpp
          POLYBENCH_JACOBI_1D-OMPTarget.cpp
          POLYBENCH_JACOBI_2D.cpp
          POLYBENCH_JACOBI_2D-Seq.cpp
          POLYBENCH_JACOBI_2D-Cuda.cpp
          POLYBENCH_JACOBI_2D-OMP.cpp
          POLYBENCH_JACOBI_2D-OMPTarget.cpp
          POLYBENCH_MVT.cpp
          POLYBENCH_MVT-Seq.cpp
          POLYBENCH_MVT-Cuda.cpp
          POLYBENCH_MVT-OMP.cpp
          POLYBENCH_MVT-OMPTarget.cpp
  DEPENDS_ON common ${RAJA_PERFSUITE_DEPENDS}
  )
