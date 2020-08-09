//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_HEAT_3D.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>


namespace rajaperf 
{
namespace polybench
{

 
void POLYBENCH_HEAT_3D::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps= getRunReps();

  POLYBENCH_HEAT_3D_DATA_SETUP;

  POLYBENCH_HEAT_3D_VIEWS_RAJA;

  auto poly_heat3d_lam1 = [=](Index_type i, Index_type j, Index_type k) {
                            POLYBENCH_HEAT_3D_BODY1_RAJA;
                          };
  auto poly_heat3d_lam2 = [=](Index_type i, Index_type j, Index_type k) {
                            POLYBENCH_HEAT_3D_BODY2_RAJA;
                          };

  switch ( vid ) {
    case RAJA_OpenMP : {
     
      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                    RAJA::ArgList<0, 1>,
            RAJA::statement::For<2, RAJA::loop_exec,
              RAJA::statement::Lambda<0>
            >
          >,
          RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                    RAJA::ArgList<0, 1>,
            RAJA::statement::For<2, RAJA::loop_exec,
              RAJA::statement::Lambda<1>
            >
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) {

          RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{1, N-1},
                                                   RAJA::RangeSegment{1, N-1},
                                                   RAJA::RangeSegment{1, N-1}),

            poly_heat3d_lam1,
            poly_heat3d_lam2
          );

        }

      }
      stopTimer();

      POLYBENCH_HEAT_3D_DATA_RESET;
      
      break;
    }

    case Hand_Opt : {
     
      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                    RAJA::ArgList<0, 1>,
            RAJA::statement::For<2, RAJA::loop_exec,
              RAJA::statement::Lambda<0>
            >
          >,
          RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                    RAJA::ArgList<0, 1>,
            RAJA::statement::For<2, RAJA::loop_exec,
              RAJA::statement::Lambda<1>
            >
          >,
          RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                    RAJA::ArgList<0, 1>,
            RAJA::statement::For<2, RAJA::loop_exec,
              RAJA::statement::Lambda<2>
            >
          >
        >;

      using KPol =
        RAJA::KernelPolicy<
          RAJA::statement::For<0,RAJA::omp_parallel_for_exec,
            RAJA::statement::For<1,RAJA::loop_exec,
              RAJA::statement::For<2,RAJA::loop_exec,
                RAJA::statement::Lambda<0>
              >
            >
          >
        >;

      auto lam1 = [=](auto i, auto j, auto k) {
        POLYBENCH_HEAT_3D_A_INTO_B;
      };
      auto lam2 = [=](auto i, auto j, auto k) {
        POLYBENCH_HEAT_3D_B_INTO_C;
      };
      auto lam3 = [=](auto i, auto j, auto k) {
        POLYBENCH_HEAT_3D_COPY_C;
      };

      startTimer();
 
     
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) {

          RAJA::kernel<KPol>( RAJA::make_tuple(RAJA::RangeSegment{1, N-1},
                                                   RAJA::RangeSegment{1, N-1},
                                                   RAJA::RangeSegment{1, N-1}),

            lam1         
          );
          RAJA::kernel<KPol>( RAJA::make_tuple(RAJA::RangeSegment{1, N-1},
                                                   RAJA::RangeSegment{1, N-1},
                                                   RAJA::RangeSegment{1, N-1}),

            lam2       
          );
          RAJA::kernel<KPol>( RAJA::make_tuple(RAJA::RangeSegment{1, N-1},
                                                   RAJA::RangeSegment{1, N-1},
                                                   RAJA::RangeSegment{1, N-1}),

            lam3         
          );

        }

      }
      stopTimer();

      POLYBENCH_HEAT_3D_DATA_RESET;
      
      break;
    }
 
   case LoopChain : {
     
        using KPol =
        RAJA::KernelPolicy<
          RAJA::statement::For<0,RAJA::omp_parallel_for_exec,
            RAJA::statement::For<1,RAJA::loop_exec,
              RAJA::statement::For<2,RAJA::loop_exec,
                RAJA::statement::Lambda<0>
              >
            >
          >
        >;

      auto lam1 = [=](auto i, auto j, auto k) {
        POLYBENCH_HEAT_3D_A_INTO_B;
      };
      auto lam2 = [=](auto i, auto j, auto k) {
        POLYBENCH_HEAT_3D_B_INTO_C;
      };
      auto lam3 = [=](auto i, auto j, auto k) {
        POLYBENCH_HEAT_3D_COPY_C;
      };

      auto seg = RAJA::make_tuple(RAJA::RangeSegment{1, N-1},
                                                   RAJA::RangeSegment{1, N-1},
                                                   RAJA::RangeSegment{1, N-1});
 
      auto knl1 = RAJA::make_kernel<KPol>(seg, lam1);
      auto knl2 = RAJA::make_kernel<KPol>(seg, lam2);
      auto knl3 = RAJA::make_kernel<KPol>(seg, lam3);

      auto tiledKnl = RAJA::overlapped_tile_no_fuse<12>(knl1,knl2);
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) {

          tiledKnl();
          knl3();
        }

      }
      stopTimer();

      POLYBENCH_HEAT_3D_DATA_RESET;
      
      break;
    }

    default : {
      std::cout << "\n  POLYBENCH_HEAT_3D : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace polybench
} // end namespace rajaperf
