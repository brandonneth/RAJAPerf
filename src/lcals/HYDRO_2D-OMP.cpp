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
          RAJA::statement::For<0, RAJA::omp_parallel_for_exec,  // k
            RAJA::statement::For<1, RAJA::loop_exec,  // j
              RAJA::statement::Lambda<0>
            >
          >
        >;

      auto kseg = RAJA::RangeSegment(kbeg, kend);
      auto jseg = RAJA::RangeSegment(jbeg, jend);

      //shifting knl 2 by (1,1)
      //shifting knl 3 by (1,1)
      auto l2 = [=](auto k, auto j) {
        zu(k-1,j-1) += s*( za(k-1,j-1) * ( zz(k-1,j-1) - zz(k-1,j) ) - 
                           za(k-1,j-2) * ( zz(k-1,j-1) - zz(k-1,j-2) ) - 
                           zb(k-1,j-1) * ( zz(k-1,j-1) - zz(k-2,j-1) ) + 
                           zb(k,j-1) * ( zz(k-1,j-1) - zz(k,j-1) ) ); 
        zv(k-1,j-1) += s*( za(k-1,j-1) * ( zr(k-1,j-1) - zr(k-1,j) ) - 
                           za(k-1,j-2) * ( zr(k-1,j-1) - zr(k-1,j-2) ) - 
                           zb(k-1,j-1) * ( zr(k-1,j-1) - zr(k-2,j-1) ) + 
                           zb(k,j-1) * ( zr(k-1,j-1) - zr(k,j-1) ) );
      };

      auto l3 = [=](auto k, auto j) {
        zrout(k-1,j-1) = zr(k-1,j-1) + t * zu(k-1,j-1);
        zzout(k-1,j-1) = zz(k-1,j-1) + t * zv(k-1,j-1);
      };

      auto kseg2 = RAJA::RangeSegment(kbeg+1,kend+1);
      auto jseg2 = RAJA::RangeSegment(jbeg+1,jend+1);
      auto kseg3 = RAJA::RangeSegment(kbeg+1,kend+1);
      auto jseg3 = RAJA::RangeSegment(jbeg+1,jend+1);
      
      auto overlapSeg = RAJA::make_tuple(RAJA::RangeSegment(kbeg+1, kend), RAJA::RangeSegment(jbeg+1, jend));


      using namespace RAJA;

      int tileSize = 32;
      int kTiles = ((kend-kbeg+1) / tileSize) + 1;
      int jTiles = ((jend-jbeg+1) / tileSize) + 1;
  
      auto tiled_lam = [=](auto kTile, auto jTile) {
        auto ks = (kTile * tileSize) + kbeg + 1;
        auto js = (jTile * tileSize) + jbeg + 1;
        
        auto ke = ((kTile + 1) * tileSize) + kbeg + 1;
        auto je = ((jTile + 1) * tileSize) + jbeg + 1;
        
        if(ke > kend) {ke = kend;}
        if(je > jend) {je = jend;}

        //overlap for knl 1 is (2,2)
        auto ks1 = ks-2;
        auto js1 = js-2;
 
        if(ks1 < kbeg+1) {ks1 = kbeg+1;}
        if(js1 < jbeg+1) {js1 = jbeg+1;}

        auto seg1 = make_tuple(RangeSegment(ks1, ke), RangeSegment(js1, je));
        auto seg2 = make_tuple(RangeSegment(ks, ke), RangeSegment(js,je));

        kernel<EXECPOL>(seg1, hydro2d_lam1);
        kernel<EXECPOL>(seg2, l2);
        kernel<EXECPOL>(seg2, l3);
      };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

          //low boundary iterations
          kernel<EXECPOL>(make_tuple(RangeSegment(kbeg,kbeg+1), RangeSegment(jbeg, jend)), hydro2d_lam1);
          kernel<EXECPOL>(make_tuple(RangeSegment(kbeg+1,kend), RangeSegment(jbeg,jbeg+1)), hydro2d_lam1);
          

          //overlap iterations
          for(int kTile = 0; kTile < kTiles; kTile++) {
            for(int jTile = 0; jTile < jTiles; jTile++) {
              //tiled_lam(kTile, jTile);
            }
          }
          kernel<EXECPOL>(make_tuple(RangeSegment(0,kTiles), RangeSegment(0,jTiles)), tiled_lam); 
          //RAJA::kernel<EXECPOL>(overlapSeg, hydro2d_lam1);
          //RAJA::kernel<EXECPOL>(overlapSeg, l2);
          //RAJA::kernel<EXECPOL>(overlapSeg, l3);

          //high boundary iterations
          kernel<EXECPOL>(make_tuple(RangeSegment(kbeg+1, kend), RangeSegment(jend, jend+1)), l2); 
          kernel<EXECPOL>(make_tuple(RangeSegment(kend, kend+1), RangeSegment(jbeg+1, jend+1)), l2); 
          
          kernel<EXECPOL>(make_tuple(RangeSegment(kbeg+1, kend), RangeSegment(jend, jend+1)), l3); 
          kernel<EXECPOL>(make_tuple(RangeSegment(kend, kend+1), RangeSegment(jbeg+1, jend+1)), l3); 


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

      auto tiledKnl = RAJA::overlapped_tile_no_fuse<32>(knl1,knl2,knl3);
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
