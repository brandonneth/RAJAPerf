//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PRESSURE.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace apps
{


void PRESSURE::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  PRESSURE_DATA_SETUP;

  auto pressure_lam1 = [=](Index_type i) {
                         PRESSURE_BODY1;
                       };
  auto pressure_lam2 = [=](Index_type i) {
                         PRESSURE_BODY2;
                       };
  
  switch ( vid ) {


    case RAJA_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::region<RAJA::omp_parallel_region>( [=]() {

          RAJA::forall<RAJA::omp_for_nowait_exec>(
            RAJA::RangeSegment(ibegin, iend), pressure_lam1);

          RAJA::forall<RAJA::omp_for_nowait_exec>(
            RAJA::RangeSegment(ibegin, iend), pressure_lam2);

        }); // end omp parallel region

      }
      stopTimer();

      break;
    }

    case Hand_Opt : {

      auto fused_lam = [=](Index_type i) {
                         PRESSURE_BODY1;
                         PRESSURE_BODY2;
                       };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
          RAJA::forall<RAJA::omp_parallel_for_exec>(
            RAJA::RangeSegment(ibegin, iend), fused_lam);
      }
      stopTimer();

      break;
    }

    case LoopChain : {
      using EPol = RAJA::omp_parallel_for_exec;
      auto seg = RAJA::RangeSegment(ibegin, iend);
      auto knl1 = RAJA::make_forall<EPol>(seg, pressure_lam1);
      auto knl2 = RAJA::make_forall<EPol>(seg, pressure_lam2);

      auto fusedKnl = RAJA::fuse(knl1,knl2);
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        fusedKnl();
      }
      stopTimer();

      break;
    }

    default : {
      std::cout << "\n  PRESSURE : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace apps
} // end namespace rajaperf
