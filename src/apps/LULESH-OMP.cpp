//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "LULESH.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace apps
{

void LULESH::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  LULESH_DATA_SETUP;
  
  auto lulesh_lam1 = [=](Index_type i) {
                       LULESH_BODY1;
                     };
  auto lulesh_lam2 = [=](Index_type i) {
                       LULESH_BODY2;
                     };
  auto lulesh_lam3 = [=](Index_type i) {
                       LULESH_BODY3;
                     };
  auto lulesh_lam4 = [=](Index_type i) {
                       LULESH_BODY4;
                     };
  auto lulesh_lam5 = [=](Index_type i) {
                       LULESH_BODY5;
                     };
  auto lulesh_lam6 = [=](Index_type i) {
                       LULESH_BODY6;
                     };
  auto lulesh_lam7 = [=](Index_type i) {
                       LULESH_BODY7;
                     };
  auto lulesh_lam8 = [=](Index_type i) {
                       LULESH_BODY8;
                     };
  auto lulesh_lam9 = [=](Index_type i) {
                       LULESH_BODY9;
                     };
  auto lulesh_lam10 = [=](Index_type i) {
                       LULESH_BODY10;
                     };
  auto lulesh_lam11 = [=](Index_type i) {
                       LULESH_BODY11;
                     };
  auto lulesh_lam12 = [=](Index_type i) {
                       LULESH_BODY12;
                     };

  switch ( vid ) {


    case RAJA_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::region<RAJA::omp_parallel_region>( [=]() {

          RAJA::forall<RAJA::omp_for_nowait_exec>(
            RAJA::RangeSegment(ibegin, iend), lulesh_lam1);

          RAJA::forall<RAJA::omp_for_nowait_exec>(
            RAJA::RangeSegment(ibegin, iend), lulesh_lam2);

          RAJA::forall<RAJA::omp_for_nowait_exec>(
            RAJA::RangeSegment(ibegin, iend), lulesh_lam3);
  
          RAJA::forall<RAJA::omp_for_nowait_exec>(
            RAJA::RangeSegment(ibegin, iend), lulesh_lam4);
  
          RAJA::forall<RAJA::omp_for_nowait_exec>(
            RAJA::RangeSegment(ibegin, iend), lulesh_lam5);
  
          RAJA::forall<RAJA::omp_for_nowait_exec>(
            RAJA::RangeSegment(ibegin, iend), lulesh_lam6);
  
          RAJA::forall<RAJA::omp_for_nowait_exec>(
            RAJA::RangeSegment(ibegin, iend), lulesh_lam7);

          RAJA::forall<RAJA::omp_for_nowait_exec>(
            RAJA::RangeSegment(ibegin, iend), lulesh_lam8);

          RAJA::forall<RAJA::omp_for_nowait_exec>(
            RAJA::RangeSegment(ibegin, iend), lulesh_lam9);
  
          RAJA::forall<RAJA::omp_for_nowait_exec>(
            RAJA::RangeSegment(ibegin, iend), lulesh_lam10);
  
          RAJA::forall<RAJA::omp_for_nowait_exec>(
            RAJA::RangeSegment(ibegin, iend), lulesh_lam11);
  
          RAJA::forall<RAJA::omp_for_nowait_exec>(
            RAJA::RangeSegment(ibegin, iend), lulesh_lam12);

        }); // end omp parallel region

      }
      stopTimer();
      break;
    }

    case Hand_Opt: {
      auto fused_lam = [=](Index_type i) {
                         LULESH_BODY1; 
                         LULESH_BODY2; 
                         LULESH_BODY3; 
                         LULESH_BODY4; 
                         LULESH_BODY5; 
                         LULESH_BODY6; 
                         LULESH_BODY7; 
                         LULESH_BODY8; 
                         LULESH_BODY9; 
                         LULESH_BODY10; 
                         LULESH_BODY11; 
                         LULESH_BODY12; 

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

      auto knl1 = RAJA::make_forall<EPol>(seg, lulesh_lam1);

      auto knl2 = RAJA::make_forall<EPol>(seg, lulesh_lam2);

      auto knl3 = RAJA::make_forall<EPol>(seg, lulesh_lam3);

      auto knl4 = RAJA::make_forall<EPol>(seg, lulesh_lam4);
      auto knl5 = RAJA::make_forall<EPol>(seg, lulesh_lam5);
      auto knl6 = RAJA::make_forall<EPol>(seg, lulesh_lam6);
     
      auto knl7 = RAJA::make_forall<EPol>(seg, lulesh_lam7);

      auto knl8 = RAJA::make_forall<EPol>(seg, lulesh_lam8);

      auto knl9 = RAJA::make_forall<EPol>(seg, lulesh_lam9);

      auto knl10 = RAJA::make_forall<EPol>(seg, lulesh_lam10);
      auto knl11 = RAJA::make_forall<EPol>(seg, lulesh_lam11);
      auto knl12 = RAJA::make_forall<EPol>(seg, lulesh_lam12);

      auto fusedKnl = RAJA::fuse(knl1,knl2,knl3,knl4,knl5,knl6,knl7, knl8, knl9, knl10, knl11, knl12);
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        fusedKnl();
      }
      stopTimer();
      break;
    }

 
    default : {
      std::cout << "\n  LULESH : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace apps
} // end namespace rajaperf
