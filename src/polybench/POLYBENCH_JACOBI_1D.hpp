//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// POLYBENCH_JACOBI_1D kernel reference implementation:
///
/// for (t = 0; t < TSTEPS; t++)
/// {
///   swap(A1,A2);
///   for (i = 1; i < N - 1; i++) {
///     B[i] = 0.33333 * (A1[i-1] + A1[i] + A1[i + 1]);
///   }
///   for (i = 1; i < N - 1; i++) {
///     A2[i] = 0.33333 * (B[i-1] + B[i] + B[i + 1]);
///   }
/// }


#ifndef RAJAPerf_POLYBENCH_JACOBI_1D_HPP
#define RAJAPerf_POLYBENCH_JACOBI_1D_HPP

#define POLYBENCH_JACOBI_1D_DATA_SETUP \
  using ViewType = RAJA::View<Real_type, RAJA::Layout<1, Index_type, 1>>; \
  ViewType A1(m_A1init, m_N);\
  ViewType B(m_Binit, m_N);\
  ViewType A2(m_A2init, m_N);\
  const Index_type N = getRunSize(); \
  const Index_type tsteps = m_tsteps;

#define POLYBENCH_JACOBI_1D_DATA_RESET \
  m_A1init = m_A1; \
  m_A2init = m_A2; \
  m_Binit = m_B; \
  m_A1 = A1.get_data(); \
  m_B = B.get_data(); \
  m_A2 = A2.get_data();


#define POLYBENCH_JACOBI_1D_BODY1 \
  B(i) = 0.33333 * (A1(i-1) + A1(i) + A1(i+1));  

#define POLYBENCH_JACOBI_1D_BODY2 \
  A2(i) = 0.33333 * (B(i-1) + B(i) + B(i+1));

#define SWAP \
  auto temp = A2.get_data(); \
  A2.set_data(A1.get_data()); \
  A1.set_data(temp); \

#include "common/KernelBase.hpp"

namespace rajaperf 
{

class RunParams;

namespace polybench
{

class POLYBENCH_JACOBI_1D : public KernelBase
{
public:

  POLYBENCH_JACOBI_1D(const RunParams& params);

  ~POLYBENCH_JACOBI_1D();


  void setUp(VariantID vid);
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);
  void runCudaVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Index_type m_N;
  Index_type m_tsteps;

  Real_ptr m_A1;
  Real_ptr m_B;
  Real_ptr m_A2;
  Real_ptr m_A1init;
  Real_ptr m_A2init;
  Real_ptr m_Binit;
};

} // end namespace polybench
} // end namespace rajaperf

#endif // closing endif for header file include guard
