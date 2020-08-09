//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HYDRO_2D.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace lcals
{


void HYDRO_2D::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  const Index_type kbeg = 1;
  const Index_type kend = m_kn - 1;
  const Index_type jbeg = 1;
  const Index_type jend = m_jn - 1;

  HYDRO_2D_DATA_SETUP;

  auto hydro2d_base_lam1 = [=] (Index_type k, Index_type j) {
                             HYDRO_2D_BODY1;
                           };
  auto hydro2d_base_lam2 = [=] (Index_type k, Index_type j) {
                             HYDRO_2D_BODY2;
                           };
  auto hydro2d_base_lam3 = [=] (Index_type k, Index_type j) {
                             HYDRO_2D_BODY3;
                           };

  HYDRO_2D_VIEWS_RAJA;

  auto hydro2d_lam1 = [=] (auto k, auto j) {
                        HYDRO_2D_BODY1_RAJA;
                      };
  auto hydro2d_lam2 = [=] (auto k, auto j) {
                        HYDRO_2D_BODY2_RAJA;
                      };
  auto hydro2d_lam3 = [=] (auto k, auto j) {
                        HYDRO_2D_BODY3_RAJA;
                      };

  
  switch ( vid ) {


    case RAJA_OpenMP : {

      using EXECPOL =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::omp_for_nowait_exec,  // k
            RAJA::statement::For<1, RAJA::loop_exec,  // j
              RAJA::statement::Lambda<0>
            >
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::region<RAJA::omp_parallel_region>( [=]() {

          RAJA::kernel<EXECPOL>(
                       RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                                         RAJA::RangeSegment(jbeg, jend)),
                       hydro2d_lam1); 

          RAJA::kernel<EXECPOL>(
                       RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                                         RAJA::RangeSegment(jbeg, jend)),
                       hydro2d_lam2); 

          RAJA::kernel<EXECPOL>(
                       RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                                         RAJA::RangeSegment(jbeg, jend)),
                       hydro2d_lam3); 

        }); // end omp parallel region 

      }
      stopTimer();

      break;
    }

    case Hand_Opt : {

      using EXECPOL =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::omp_for_nowait_exec,  // k
            RAJA::statement::For<1, RAJA::loop_exec,  // j
              RAJA::statement::Lambda<0>
            >
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::region<RAJA::omp_parallel_region>( [=]() {

          RAJA::kernel<EXECPOL>(
                       RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                                         RAJA::RangeSegment(jbeg, jend)),
                       hydro2d_lam1); 

          RAJA::kernel<EXECPOL>(
                       RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                                         RAJA::RangeSegment(jbeg, jend)),
                       hydro2d_lam2); 

          RAJA::kernel<EXECPOL>(
                       RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                                         RAJA::RangeSegment(jbeg, jend)),
                       hydro2d_lam3); 

        }); // end omp parallel region 

      }
      stopTimer();

      break;
    }

  
    case LoopChain : {

      using EXECPOL =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::omp_for_nowait_exec,  // k
            RAJA::statement::For<1, RAJA::loop_exec,  // j
              RAJA::statement::Lambda<0>
            >
          >
        >;
      auto seg = RAJA::make_tuple(RAJA::RangeSegment(kbeg, kend), RAJA::RangeSegment(jbeg, jend));

      auto knl1 = RAJA::make_kernel<EXECPOL>(seg, hydro2d_lam1);
      auto knl2 = RAJA::make_kernel<EXECPOL>(seg, hydro2d_lam2);
      auto knl3 = RAJA::make_kernel<EXECPOL>(seg, hydro2d_lam3);

      auto tiledKnl = overlapped_tile_no_fuse(knl1,knl2,knl3);
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       tiledKnl();
      }
      stopTimer();

      break;
    }

    default : {
      std::cout << "\n  HYDRO_2D : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace lcals
} // end namespace rajaperf
