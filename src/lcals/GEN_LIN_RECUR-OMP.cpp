//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "GEN_LIN_RECUR.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace lcals
{


void GEN_LIN_RECUR::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();

  GEN_LIN_RECUR_DATA_SETUP;

  auto genlinrecur_lam1 = [=](auto k) {
                            GEN_LIN_RECUR_RAJA_BODY1;
                          };
  auto genlinrecur_lam2 = [=](auto i) {
                            GEN_LIN_RECUR_RAJA_BODY2;
                          };

  switch ( vid ) {


    case RAJA_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(0, N), genlinrecur_lam1);

        RAJA::forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(1, N+1), genlinrecur_lam2);

      }
      stopTimer();

      break;
    }

    case Hand_Opt : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(0, N),  [=](auto k) {
                            GEN_LIN_RECUR_RAJA_BODY1;
                            GEN_LIN_RECUR_RAJA_BODY1;
                          });

      }
      stopTimer();

      break;
    }

   
    case LoopChain : {
      auto knl1 = RAJA::make_forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(0, N), genlinrecur_lam1);
      auto knl2 = RAJA::make_forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(1, N+1), genlinrecur_lam1);

      auto fusedKnl = RAJA::fuse(knl1, knl2);

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        fusedKnl();
        
      }
      stopTimer();

      break;
    }

    default : {
      std::cout << "\n  GEN_LIN_RECUR : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace lcals
} // end namespace rajaperf
