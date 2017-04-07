/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for Basic kernel IF_QUAD.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-xxxxxx
//
// All rights reserved.
//
// This file is part of the RAJA Performance Suite.
//
// For additional details, please read the file LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#ifndef RAJAPerf_Basic_IF_QUAD_HXX
#define RAJAPerf_Basic_IF_QUAD_HXX

#include "common/KernelBase.hxx"

namespace rajaperf 
{
class RunParams;

namespace basic
{

class IF_QUAD : public KernelBase
{
public:

  IF_QUAD(const RunParams& params);

  ~IF_QUAD();

  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

private:
  Real_ptr m_a;
  Real_ptr m_b;
  Real_ptr m_c;
  Real_ptr m_x1;
  Real_ptr m_x2;
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
