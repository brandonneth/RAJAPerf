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
                 POLYBENCH_JACOBI_1D_BODY1;
               };
  auto lam2 =  [=] (auto i) {
                 POLYBENCH_JACOBI_1D_BODY2;
               };
 
  switch ( vid ) {

    case RAJA_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) {
          
          SWAP;

          RAJA::forall<RAJA::omp_parallel_for_exec> (RAJA::RangeSegment{1, N-1},
            lam1
          );

          RAJA::forall<RAJA::omp_parallel_for_exec> (RAJA::RangeSegment{1, N-1},
            lam2
          );
        
        }

      }
      stopTimer();

      POLYBENCH_JACOBI_1D_DATA_RESET;

      break;
    }
    case Hand_Opt: {
      startTimer();

      auto initialSeg = RAJA::RangeSegment{1,N-1};
      
      //we shift the initial segment forward 1 for the second loop
      auto shiftedSeg = RAJA::RangeSegment{2,N};

      auto lam2_shifted = [=](auto i) {
        A2(i-1) = 0.33333 * ( B(i-2) + B(i-1) + B(i) );
      };
      
      auto overlapSeg = RAJA::RangeSegment(2,N-1);
      
      int tileSize = 2048;
      int numTiles = ((N-3) / tileSize) + 1;
      auto tiledLam = [=](int tileNum) {
        int start = tileNum*tileSize + 2;
        int end = (tileNum+1) * tileSize + 2;
      
        int start1 = start - 2;
        
        if(start1 < 2) { start1 = 2;} 
        if(end > N-1) { end = N -1;}
        auto tileSeg1 = RAJA::RangeSegment(start1, end);
        auto tileSeg2 = RAJA::RangeSegment(start, end);
        RAJA::forall<RAJA::loop_exec>(tileSeg1, lam1);
        RAJA::forall<RAJA::loop_exec>(tileSeg2, lam2_shifted);
 
      };
      
   
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) {

          SWAP;
          //pre-overlap
          lam1(1);

          RAJA::forall<RAJA::omp_parallel_for_exec>(RAJA::RangeSegment(0,numTiles), tiledLam);
          for(int tileNum = 0; tileNum < numTiles; tileNum++) {
            //tiledLam(tileNum);
           
          }
          // RAJA::forall<RAJA::omp_parallel_for_exec> (overlapSeg, lam1);

          //RAJA::forall<RAJA::omp_parallel_for_exec> (overlapSeg, lam2_shifted);
        
          lam2_shifted(N-1);
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
           
      auto tiledKnl = RAJA::overlapped_tile_no_fuse<2048*2*2*2*2>(knl1,knl2);
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) {
          SWAP;

          tiledKnl();
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
