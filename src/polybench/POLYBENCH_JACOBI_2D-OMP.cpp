//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_JACOBI_2D.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>


namespace rajaperf 
{
namespace polybench
{


void POLYBENCH_JACOBI_2D::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps= getRunReps();

  POLYBENCH_JACOBI_2D_DATA_SETUP;

  auto poly_jacobi2d_base_lam1 = [=](Index_type i, Index_type j) {
                                   POLYBENCH_JACOBI_2D_BODY1;
                                 };
  auto poly_jacobi2d_base_lam2 = [=](Index_type i, Index_type j) {
                                   POLYBENCH_JACOBI_2D_BODY2;
                                 };

  POLYBENCH_JACOBI_2D_VIEWS_RAJA;

  auto poly_jacobi2d_lam1 = [=](Index_type i, Index_type j) {
                              POLYBENCH_JACOBI_2D_BODY1_RAJA;
                            };
  auto poly_jacobi2d_lam2 = [=](Index_type i, Index_type j) {
                              POLYBENCH_JACOBI_2D_BODY2_RAJA;
                            };

  auto lam1 = [=](auto i, auto j) {
   POLYBENCH_JACOBI_2D_A2B;
  };
  auto lam2 = [=](auto i, auto j) {
   POLYBENCH_JACOBI_2D_B2C;
  };
  auto lam3 = [=](auto i, auto j) {
   POLYBENCH_JACOBI_2D_C2A;
  };

  switch ( vid ) {


    case RAJA_OpenMP : {

      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::Lambda<0>
            >
          >,
          RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::Lambda<1>
            >
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) {

          RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{1, N-1},
                                                   RAJA::RangeSegment{1, N-1}),

	    poly_jacobi2d_lam1,
	    poly_jacobi2d_lam2
          );

        }

      }
      stopTimer();

      POLYBENCH_JACOBI_2D_DATA_RESET;

      break;
    }

    case Hand_Opt: {

      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::Lambda<0>
            >
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) {

          RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{1, N-1},
                                                   RAJA::RangeSegment{1, N-1}),

	    lam1
          );
          RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{1, N-1},
                                                   RAJA::RangeSegment{1, N-1}),

	    lam2
          );
          RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{1, N-1},
                                                   RAJA::RangeSegment{1, N-1}),

	    lam3
          );


        }

      }
      stopTimer();

      POLYBENCH_JACOBI_2D_DATA_RESET;

      break;
    }

    case LC_Tiled : {

      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::Lambda<0>
            >
          >
        >;
  
      auto seg =  RAJA::make_tuple(RAJA::RangeSegment{1, N-1},
                                   RAJA::RangeSegment{1, N-1});
      auto knl1 = RAJA::make_kernel<EXEC_POL>(seg, lam1);
      auto knl2 = RAJA::make_kernel<EXEC_POL>(seg, lam2);
      auto knl3 = RAJA::make_kernel<EXEC_POL>(seg, lam3);
     
      auto tiledKnl = RAJA::overlapped_tile_no_fuse<32>(knl1, knl2);
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) {
          tiledKnl();
          knl3();
         
        }

      }
      stopTimer();

      POLYBENCH_JACOBI_2D_DATA_RESET;

      break;
    }

    case LC_Fused : {

      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::Lambda<0>
            >
          >,
          RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::Lambda<1>
            >
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) {

          RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{1, N-1},
                                                   RAJA::RangeSegment{1, N-1}),

	    poly_jacobi2d_lam1,
	    poly_jacobi2d_lam2
          );

        }

      }
      stopTimer();

      POLYBENCH_JACOBI_2D_DATA_RESET;

      break;
    }

    default : {
      std::cout << "\n  POLYBENCH_JACOBI_2D : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace polybench
} // end namespace rajaperf
