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
///   for (i = 1; i < N - 1; i++) {
///     B[i] = 0.33333 * (A[i-1] + A[i] + A[i + 1]);
///   }
///   for (i = 1; i < N - 1; i++) {
///     A[i] = 0.33333 * (B[i-1] + B[i] + B[i + 1]);
///   }
/// }


#ifndef RAJAPerf_POLYBENCH_JACOBI_1D_HPP
#define RAJAPerf_POLYBENCH_JACOBI_1D_HPP

#define POLYBENCH_JACOBI_1D_DATA_SETUP \
  using ViewType = RAJA::View<Real_type, RAJA::Layout<1, Index_type, 1>>; \
  ViewType A(m_Ainit, m_N);\
  ViewType B(m_Binit, m_N);\
  ViewType C(m_Cinit, m_N);\
  const Index_type N = getRunSize(); \
  const Index_type tsteps = m_tsteps;

#define POLYBENCH_JACOBI_1D_DATA_RESET \
  m_Ainit = m_A; \
  m_Binit = m_B; \
  m_Cinit = m_C; \
  m_A = A.data; \
  m_B = B.data; \
  m_C = C.data;


#define POLYBENCH_JACOBI_1D_BODY1 \
  B(i) = 0.33333 * (A(i-1) + A(i) + A(i+1));  

#define POLYBENCH_JACOBI_1D_BODY2 \
  A(i) = 0.33333 * (B(i-1) + B(i) + B(i+1));

#define POLYBENCH_JACOBI_1D_A2B \
  B(i) = 0.33333 * (A(i-1) + A(i) + A(i+1));

#define POLYBENCH_JACOBI_1D_B2C \
  C(i) = 0.33333 * (B(i-1) + B(i) + B(i+1));

#define POLYBENCH_JACOBI_1D_C2A \
  A(i) = C(i);

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

  Real_ptr m_A;
  Real_ptr m_B;
  Real_ptr m_C;
  Real_ptr m_Ainit;
  Real_ptr m_Binit;
  Real_ptr m_Cinit;
};

} // end namespace polybench
} // end namespace rajaperf

#endif // closing endif for header file include guard
