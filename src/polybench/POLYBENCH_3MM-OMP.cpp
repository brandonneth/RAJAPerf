//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_3MM.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>
#include <cstring>


#define USE_OMP_COLLAPSE
//#undef USE_OMP_COLLAPSE

#define USE_RAJA_OMP_COLLAPSE
//#undef USE_RAJA_OMP_COLLAPSE


namespace rajaperf 
{
namespace polybench
{

  
void POLYBENCH_3MM::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();

  POLYBENCH_3MM_DATA_SETUP;

  POLYBENCH_3MM_VIEWS_RAJA;

  auto poly_3mm_lam1 = [=] (Index_type /*i*/, Index_type /*j*/, Index_type /*k*/,
                            Real_type &dot) {
                              POLYBENCH_3MM_BODY1_RAJA;
                            };
  auto poly_3mm_lam2 = [=] (Index_type i, Index_type j, Index_type k, 
                            Real_type &dot) {
                              POLYBENCH_3MM_BODY2_RAJA;
                            };
  auto poly_3mm_lam3 = [=] (Index_type i, Index_type j, Index_type /*k*/, 
                            Real_type &dot) {
                              POLYBENCH_3MM_BODY3_RAJA;
                            };

  auto poly_3mm_lam4 = [=] (Index_type /*j*/, Index_type /*l*/, Index_type /*m*/,
                            Real_type &dot) {
                              POLYBENCH_3MM_BODY4_RAJA;
                            };
  auto poly_3mm_lam5 = [=] (Index_type j, Index_type l, Index_type m, 
                            Real_type &dot) {
                              POLYBENCH_3MM_BODY5_RAJA;
                            };
  auto poly_3mm_lam6 = [=] (Index_type j, Index_type l, Index_type /*m*/, 
                            Real_type &dot) {
                              POLYBENCH_3MM_BODY6_RAJA;
                            };
  auto poly_3mm_lam7 = [=] (Index_type /*i*/, Index_type /*l*/, Index_type /*j*/,
                            Real_type &dot) {
                              POLYBENCH_3MM_BODY7_RAJA;
                            };
  auto poly_3mm_lam8 = [=] (Index_type i, Index_type l, Index_type j, 
                            Real_type &dot) {
                              POLYBENCH_3MM_BODY8_RAJA;
                            };
  auto poly_3mm_lam9 = [=] (Index_type i, Index_type l, Index_type /*j*/, 
                            Real_type &dot) {
                              POLYBENCH_3MM_BODY9_RAJA;
                            };

  switch ( vid ) {

    case RAJA_OpenMP : {

      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::Lambda<0>,
              RAJA::statement::For<2, RAJA::loop_exec,
                RAJA::statement::Lambda<1>
              >,
              RAJA::statement::Lambda<2>
            >
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::kernel_param<EXEC_POL>(
          RAJA::make_tuple(RAJA::RangeSegment{0, ni},
                           RAJA::RangeSegment{0, nj},
                           RAJA::RangeSegment{0, nk}),
          RAJA::make_tuple(static_cast<Real_type>(0.0)),

          poly_3mm_lam1,
          poly_3mm_lam2,
          poly_3mm_lam3

        );

        RAJA::kernel_param<EXEC_POL>(
          RAJA::make_tuple(RAJA::RangeSegment{0, nj},
                           RAJA::RangeSegment{0, nl},
                           RAJA::RangeSegment{0, nm}),
          RAJA::make_tuple(static_cast<Real_type>(0.0)),

          poly_3mm_lam4,
          poly_3mm_lam5,
          poly_3mm_lam6

        );

        RAJA::kernel_param<EXEC_POL>(
          RAJA::make_tuple(RAJA::RangeSegment{0, ni},
                           RAJA::RangeSegment{0, nl},
                           RAJA::RangeSegment{0, nj}),
          RAJA::make_tuple(static_cast<Real_type>(0.0)),

          poly_3mm_lam7,
          poly_3mm_lam8,
          poly_3mm_lam9

        );

      }
      stopTimer();

      break;
    }

    case Hand_Opt : {

      using InitKPol = RAJA::KernelPolicy<
        RAJA::statement::For<0,RAJA::omp_parallel_for_exec,
          RAJA::statement::For<1,RAJA::simd_exec,
            RAJA::statement::Lambda<0>
          >
        >
      >;

      using CalcKPol =  RAJA::KernelPolicy<
        RAJA::statement::For<0,RAJA::omp_parallel_for_exec,
          RAJA::statement::For<1,RAJA::loop_exec,
            RAJA::statement::For<2,RAJA::loop_exec,
              RAJA::statement::Lambda<0>
            >
          >
        >
      >;

      auto init_lam1 = [=](auto i, auto j) {Eview(i,j) = 0.0;};
      auto init_lam2 = [=](auto i, auto j) {Fview(i,j) = 0.0;};
      auto init_lam3 = [=](auto i, auto j) {Gview(i,j) = 0.0;};

      auto calc_lam1 = [=](auto i, auto j, auto k) {Eview(i,j) += Aview(i,k) * Bview(k,j);}; 
      auto calc_lam2 = [=](auto j, auto l, auto m) {Fview(j,l) += Cview(j,m) * Dview(m,l);} ;
      auto calc_lam3 = [=](auto i, auto l, auto j) {Gview(i,l) += Eview(i,j) * Fview(j,l);} ;
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        auto init_seg1 = RAJA::make_tuple(RAJA::RangeSegment(0,ni), RAJA::RangeSegment(0,nj));
        auto init_seg2 = RAJA::make_tuple(RAJA::RangeSegment(0,nj), RAJA::RangeSegment(0,nl));
        auto init_seg3 = RAJA::make_tuple(RAJA::RangeSegment(0,ni), RAJA::RangeSegment(0,nl));

        RAJA::kernel<InitKPol>(init_seg1, init_lam1);
        RAJA::kernel<InitKPol>(init_seg2, init_lam2);
        RAJA::kernel<InitKPol>(init_seg3, init_lam3);

        auto calc_seg1 = RAJA::make_tuple(RAJA::RangeSegment{0, ni},
                           RAJA::RangeSegment{0, nj},
                           RAJA::RangeSegment{0, nk});

        auto calc_seg2 = RAJA::make_tuple(RAJA::RangeSegment{0, nj},
                           RAJA::RangeSegment{0, nl},
                           RAJA::RangeSegment{0, nm});
        
        auto calc_seg3 =  RAJA::make_tuple(RAJA::RangeSegment{0, ni},
                           RAJA::RangeSegment{0, nl},
                           RAJA::RangeSegment{0, nj});

        RAJA::kernel<CalcKPol>(calc_seg1, calc_lam1);
        RAJA::kernel<CalcKPol>(calc_seg2, calc_lam2);
        RAJA::kernel<CalcKPol>(calc_seg3, calc_lam3);

      }
      stopTimer();

      break;
    }

    case LoopChain : 
    {
      using InitKPol = RAJA::KernelPolicy<
        RAJA::statement::For<0,RAJA::omp_parallel_for_exec,
          RAJA::statement::For<1,RAJA::simd_exec,
            RAJA::statement::Lambda<0>
          >
        >
      >;

      using CalcKPol =  RAJA::KernelPolicy<
        RAJA::statement::For<0,RAJA::omp_parallel_for_exec,
          RAJA::statement::For<1,RAJA::loop_exec,
            RAJA::statement::For<2,RAJA::loop_exec,
              RAJA::statement::Lambda<0>
            >
          >
        >
      >;

      auto init_lam1 = [=](auto i, auto j) {Eview(i,j) = 0.0;};
      auto init_lam2 = [=](auto i, auto j) {Fview(i,j) = 0.0;};
      auto init_lam3 = [=](auto i, auto j) {Gview(i,j) = 0.0;};
      
      auto init_seg1 = RAJA::make_tuple(RAJA::RangeSegment(0,ni), RAJA::RangeSegment(0,nj));
      auto init_seg2 = RAJA::make_tuple(RAJA::RangeSegment(0,nj), RAJA::RangeSegment(0,nl));
      auto init_seg3 = RAJA::make_tuple(RAJA::RangeSegment(0,ni), RAJA::RangeSegment(0,nl));

      auto calc_lam1 = [=](auto i, auto j, auto k) {Eview(i,j) += Aview(i,k) * Bview(k,j);}; 
      auto calc_lam2 = [=](auto j, auto l, auto m) {Fview(j,l) += Cview(j,m) * Dview(m,l);} ;
      auto calc_lam3 = [=](auto i, auto l, auto j) {Gview(i,l) += Eview(i,j) * Fview(j,l);} ;
 
      auto calc_seg1 = RAJA::make_tuple(RAJA::RangeSegment{0, ni},
                           RAJA::RangeSegment{0, nj},
                           RAJA::RangeSegment{0, nk});

      auto calc_seg2 = RAJA::make_tuple(RAJA::RangeSegment{0, nj},
                           RAJA::RangeSegment{0, nl},
                           RAJA::RangeSegment{0, nm});
        
      auto calc_seg3 =  RAJA::make_tuple(RAJA::RangeSegment{0, ni},
                           RAJA::RangeSegment{0, nl},
                           RAJA::RangeSegment{0, nj});

      auto knl1 = RAJA::make_kernel<CalcKPol>(calc_seg1, calc_lam1);
      auto knl2 = RAJA::make_kernel<CalcKPol>(calc_seg2, calc_lam2);
      auto knl3 = RAJA::make_kernel<CalcKPol>(calc_seg3, calc_lam3);

      auto fusedKnl = fuse(knl1,knl2);
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        
        RAJA::kernel<InitKPol>(init_seg1, init_lam1);
        RAJA::kernel<InitKPol>(init_seg2, init_lam2);
        RAJA::kernel<InitKPol>(init_seg3, init_lam3);

        fusedKnl();
        knl3();
      }
      stopTimer();

      break;
      }

    
     

    

    default : {
      std::cout << "\n  POLYBENCH_2MM : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace basic
} // end namespace rajaperf
