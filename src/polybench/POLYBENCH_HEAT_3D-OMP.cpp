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
constexpr static int tilesize = 60;
 
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

  auto lam1 = [=](auto i, auto j, auto k) {
    POLYBENCH_HEAT_3D_A2B;
  };

  auto lam2 = [=](auto i, auto j, auto k) {
    POLYBENCH_HEAT_3D_B2C;
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
          SWAP_HEAT;

          RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{1, N-1},
                                                   RAJA::RangeSegment{1, N-1},
                                                   RAJA::RangeSegment{1, N-1}),

          lam1, lam2);

        }

      }
      stopTimer();

      POLYBENCH_HEAT_3D_DATA_RESET;
      
      break;
    }

    case Hand_Opt : {
     
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
      auto initialSeg = RAJA::make_tuple(RAJA::RangeSegment{1,N-1},
        RAJA::RangeSegment{1,N-1},
       RAJA::RangeSegment{1,N-1});

      //shift knl2 by 1,1,1
      auto lam2_shifted = [=](auto i, auto j, auto k) {
        Cview(i-1,j-1,k-1) = 
          0.125 * (Bview(i,j-1,k-1) - 2.0*Bview(i-1,j-1,k-1) + Bview(i-2,j-1,k-1)) + 
          0.125 * (Bview(i-1,j,k-1) - 2.0*Bview(i-1,j-1,k-1) + Bview(i-1,j-2,k-1)) + 
          0.125 * (Bview(i-1,j-1,k) - 2.0*Bview(i-1,j-1,k-1) + Bview(i-1,j-1,k-2)) +
          Bview(i-1,j-1,k-1);
      };
      
      auto shiftedSeg = RAJA::make_tuple(RAJA::RangeSegment{2,N}, RAJA::RangeSegment(2,N), RAJA::RangeSegment(2,N));

      auto overlapSeg = RAJA::make_tuple(RAJA::RangeSegment(2,N-1),RAJA::RangeSegment(2,N-1),RAJA::RangeSegment(2,N-1));
      auto tileSize = tilesize;
      auto numTiles = ((N-3) / tileSize) + 1;

      auto tiled_lam = [=](auto iTile, auto jTile, auto kTile) {
                int is = (iTile * tileSize) + 2;
                int ie = (1+iTile) * (tileSize) + 2;

                int js = jTile * tileSize + 2;
                int je = (1+jTile) * (tileSize) + 2;
       
                int ks = kTile * tileSize + 2;
                int ke = (1+kTile) * (tileSize ) + 2;

                if(ie > N-1) {ie = N-1;}
                if(je > N-1) {je = N-1;}
                if(ke > N-1) {ke = N-1;}

                int is1 = is - 2;
                int js1 = js - 2;
                int ks1 = ks - 2;
              
                if(is1 < 2) {is1 = 2;}
                if(js1 < 2) {js1 = 2;}
                if(ks1 < 2) {ks1 = 2;}

                auto seg1 = RAJA::make_tuple(RAJA::RangeSegment(is1,ie), RAJA::RangeSegment(js1,je), RAJA::RangeSegment(ks1,ke));
                auto seg2 = RAJA::make_tuple(RAJA::RangeSegment(is,ie), RAJA::RangeSegment(js,je), RAJA::RangeSegment(ks,ke));
                RAJA::kernel<KPol>(seg1, lam1);
                RAJA::kernel<KPol>(seg2, lam2_shifted);
      };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) {

          SWAP_HEAT;
          //low boundary segments
          RAJA::kernel<KPol>(RAJA::make_tuple(RAJA::RangeSegment(1,2), RAJA::RangeSegment(1,N-1), RAJA::RangeSegment(1,N-1)), lam1);
          RAJA::kernel<KPol>(RAJA::make_tuple(RAJA::RangeSegment(2,N-1), RAJA::RangeSegment(1,2), RAJA::RangeSegment(1,N-1)), lam1);
          RAJA::kernel<KPol>(RAJA::make_tuple(RAJA::RangeSegment(2,N-1), RAJA::RangeSegment(2,N-1), RAJA::RangeSegment(1,2)), lam1);  
          
          RAJA::kernel<KPol>(make_tuple(RAJA::RangeSegment(0,numTiles), RAJA::RangeSegment(0,numTiles), RAJA::RangeSegment(0,numTiles)), tiled_lam);
      
          //high boundary segments
          RAJA::kernel<KPol>(RAJA::make_tuple(RAJA::RangeSegment(2,N-1), RAJA::RangeSegment(2,N-1), RAJA::RangeSegment(N-1,N)), lam2_shifted);
          RAJA::kernel<KPol>(RAJA::make_tuple(RAJA::RangeSegment(2,N-1), RAJA::RangeSegment(N-1,N), RAJA::RangeSegment(2,N)), lam2_shifted);
          RAJA::kernel<KPol>(RAJA::make_tuple(RAJA::RangeSegment(N-1,N), RAJA::RangeSegment(2,N), RAJA::RangeSegment(2,N)), lam2_shifted);

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

      auto seg = RAJA::make_tuple(RAJA::RangeSegment{1, N-1},
                                                   RAJA::RangeSegment{1, N-1},
                                                   RAJA::RangeSegment{1, N-1});
 
      auto knl1 = RAJA::make_kernel<KPol>(seg, lam1);
      auto knl2 = RAJA::make_kernel<KPol>(seg, lam2);

      auto tiledKnl = RAJA::overlapped_tile_no_fuse<tilesize>(knl1,knl2);
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) {
          SWAP_HEAT;
          tiledKnl();
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
