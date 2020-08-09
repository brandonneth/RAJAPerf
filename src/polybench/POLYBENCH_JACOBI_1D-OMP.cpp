//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_JACOBI_1D.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>


namespace rajaperf 
{
namespace polybench
{

 
void POLYBENCH_JACOBI_1D::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps= getRunReps();

  POLYBENCH_JACOBI_1D_DATA_SETUP;

  auto poly_jacobi1d_lam1 = [=] (Index_type i) {
                              POLYBENCH_JACOBI_1D_BODY1;
                            };
  auto poly_jacobi1d_lam2 = [=] (Index_type i) {
                              POLYBENCH_JACOBI_1D_BODY2;
                            };

  auto lam1 =  [=] (auto i) {
                 POLYBENCH_JACOBI_1D_A2B;
               };
  auto lam2 =  [=] (auto i) {
                 POLYBENCH_JACOBI_1D_B2C;
               };
  auto lam3 =  [=] (auto i) {
                 POLYBENCH_JACOBI_1D_C2A;
               };

  switch ( vid ) {

    case RAJA_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) {

          RAJA::forall<RAJA::omp_parallel_for_exec> (RAJA::RangeSegment{1, N-1},
            poly_jacobi1d_lam1
          );

          RAJA::forall<RAJA::omp_parallel_for_exec> (RAJA::RangeSegment{1, N-1},
            poly_jacobi1d_lam2
          );

        }

      }
      stopTimer();

      POLYBENCH_JACOBI_1D_DATA_RESET;

      break;
    }
    case Hand_Opt: {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) {

          RAJA::forall<RAJA::omp_parallel_for_exec> (RAJA::RangeSegment{1, N-1},
            lam1
          );

          RAJA::forall<RAJA::omp_parallel_for_exec> (RAJA::RangeSegment{1, N-1},
            lam2
          );
        
          RAJA::forall<RAJA::omp_parallel_for_exec> (RAJA::RangeSegment{1, N-1},
            lam3
          );

        }

      }
      stopTimer();

      POLYBENCH_JACOBI_1D_DATA_RESET;

      break;
    }   
    case LoopChain : {
      auto seg = RAJA::RangeSegment(1,N-1);
      auto knl1 = RAJA::make_forall<RAJA::omp_parallel_for_exec> (seg, lam1);
      auto knl2 = RAJA::make_forall<RAJA::omp_parallel_for_exec> (seg, lam2);
      auto knl3 = RAJA::make_forall<RAJA::omp_parallel_for_exec> (seg, lam3);
           
      auto tiledKnl = RAJA::overlapped_tile_no_fuse<512*2*2*2>(knl1,knl2);
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) {

          tiledKnl();
          knl3();
        }

      }
      stopTimer();

      POLYBENCH_JACOBI_1D_DATA_RESET;

      break;
    }

    default : {
      std::cout << "\n  POLYBENCH_JACOBI_1D : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace polybench
} // end namespace rajaperf
