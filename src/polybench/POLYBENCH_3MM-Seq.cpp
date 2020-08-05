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


namespace rajaperf 
{
namespace polybench
{

  
void POLYBENCH_3MM::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_3MM_DATA_SETUP;

  auto poly_3mm_base_lam2 = [=] (Index_type i, Index_type j, Index_type k,
                                 Real_type &dot) {
                              POLYBENCH_3MM_BODY2;
                            };
  auto poly_3mm_base_lam3 = [=] (Index_type i, Index_type j,
                                 Real_type &dot) {
                              POLYBENCH_3MM_BODY3;
                            };
  auto poly_3mm_base_lam5 = [=] (Index_type j, Index_type l, Index_type m,
                                 Real_type &dot) {
                               POLYBENCH_3MM_BODY5;
                            };
  auto poly_3mm_base_lam6 = [=] (Index_type j, Index_type l,
                                 Real_type &dot) {
                              POLYBENCH_3MM_BODY6;
                            };
  auto poly_3mm_base_lam8 = [=] (Index_type i, Index_type l, Index_type j,
                                 Real_type &dot) {
                              POLYBENCH_3MM_BODY8;
                            };
  auto poly_3mm_base_lam9 = [=] (Index_type i, Index_type l,
                                 Real_type &dot) {
                              POLYBENCH_3MM_BODY9;
                            };

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

  

  

}

} // end namespace basic
} // end namespace rajaperf
