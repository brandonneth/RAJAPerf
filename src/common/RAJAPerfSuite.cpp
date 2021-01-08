//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJAPerfSuite.hpp"

#include "RunParams.hpp"

//
// Lcals kernels...
//
#include "lcals/GEN_LIN_RECUR.hpp"
#include "lcals/HYDRO_2D.hpp"

//
// Polybench kernels...
//
#include "polybench/POLYBENCH_HEAT_3D.hpp"
#include "polybench/POLYBENCH_JACOBI_1D.hpp"
#include "polybench/POLYBENCH_JACOBI_2D.hpp"
#include "polybench/POLYBENCH_FDTD_2D.hpp"

//
// Apps kernels...
//
#include "apps/ENERGY.hpp"
#include "apps/PRESSURE.hpp"
#include "apps/LULESH.hpp"

#include <iostream>

namespace rajaperf
{

/*!
 *******************************************************************************
 *
 * \brief Array of names for each GROUP in suite.
 *
 * IMPORTANT: This is only modified when a group is added or removed.
 *
 *            IT MUST BE KEPT CONSISTENT (CORRESPONDING ONE-TO-ONE) WITH
 *            ENUM OF GROUP IDS IN HEADER FILE!!!
 *
 *******************************************************************************
 */
static const std::string GroupNames [] =
{
  std::string("Lcals"),
  std::string("Polybench"),
  std::string("Apps"),

  std::string("Unknown Group")  // Keep this at the end and DO NOT remove....

}; // END GroupNames


/*!
 *******************************************************************************
 *
 * \brief Array of names for each KERNEL in suite.
 *
 * IMPORTANT: This is only modified when a kernel is added or removed.
 *
 *            IT MUST BE KEPT CONSISTENT (CORRESPONDING ONE-TO-ONE) WITH
 *            ENUM OF KERNEL IDS IN HEADER FILE!!!
 *
 *******************************************************************************
 */
static const std::string KernelNames [] =
{
//
// Lcals kernels...
//
  std::string("Lcals_GEN_LIN_RECUR"),
  std::string("Lcals_HYDRO_2D"),
//
// Polybench kernels...
//
  std::string("Polybench_HEAT_3D"),
  std::string("Polybench_JACOBI_1D"),
  std::string("Polybench_JACOBI_2D"),
  std::string("Polybench_FDTD_2D"),
  

//
// Apps kernels...
//
  std::string("Apps_ENERGY"),
  std::string("Apps_PRESSURE"),
  std::string("Apps_LULESH"),

  std::string("Unknown Kernel")  // Keep this at the end and DO NOT remove....

}; // END KernelNames


/*!
 *******************************************************************************
 *
 * \brief Array of names for each VARIANT in suite.
 *
 * IMPORTANT: This is only modified when a new kernel is added to the suite.
 *
 *            IT MUST BE KEPT CONSISTENT (CORRESPONDING ONE-TO-ONE) WITH
 *            ENUM OF VARIANT IDS IN HEADER FILE!!!
 *
 *******************************************************************************
 */
static const std::string VariantNames [] =
{


#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
  std::string("RAJA_OpenMP"),
  std::string("Hand_Opt"),
  std::string("LoopChain"),
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)  
  std::string("Base_OMPTarget"),
  std::string("RAJA_OMPTarget"),
#endif

#if defined(RAJA_ENABLE_CUDA)
  std::string("Base_CUDA"),
  std::string("RAJA_CUDA"),
#endif

  std::string("Unknown Variant")  // Keep this at the end and DO NOT remove....

}; // END VariantNames


/*
 *******************************************************************************
 *
 * \brief Return group name associated with GroupID enum value.
 *
 *******************************************************************************
 */
const std::string& getGroupName(GroupID sid)
{
  return GroupNames[sid];
}


/*
 *******************************************************************************
 *
 * \brief Return kernel name associated with KernelID enum value.
 *
 *******************************************************************************
 */
std::string getKernelName(KernelID kid)
{
  std::string::size_type pos = KernelNames[kid].find("_");
  std::string kname(KernelNames[kid].substr(pos+1, std::string::npos));
  return kname;
}


/*
 *******************************************************************************
 *
 * \brief Return full kernel name associated with KernelID enum value.
 *
 *******************************************************************************
 */
const std::string& getFullKernelName(KernelID kid)
{
  return KernelNames[kid];
}


/*
 *******************************************************************************
 *
 * \brief Return variant name associated with VariantID enum value.
 *
 *******************************************************************************
 */
const std::string& getVariantName(VariantID vid)
{
  return VariantNames[vid];
}

/*
 *******************************************************************************
 *
 * \brief Construct and return kernel object for given KernelID enum value.
 *
 *******************************************************************************
 */
KernelBase* getKernelObject(KernelID kid,
                            const RunParams& run_params)
{
  KernelBase* kernel = 0;

  switch ( kid ) {


//
// Lcals kernels...
//
    case Lcals_GEN_LIN_RECUR : {
       kernel = new lcals::GEN_LIN_RECUR(run_params);
       break;
    }
    case Lcals_HYDRO_2D : {
       kernel = new lcals::HYDRO_2D(run_params);
       break;
    }

//
// Polybench kernels...
//
    case Polybench_HEAT_3D : {
       kernel = new polybench::POLYBENCH_HEAT_3D(run_params);
       break;
    }
    case Polybench_JACOBI_1D : {
       kernel = new polybench::POLYBENCH_JACOBI_1D(run_params);
       break;
    }
    case Polybench_JACOBI_2D : {
       kernel = new polybench::POLYBENCH_JACOBI_2D(run_params);
       break;
    }
    case Polybench_FDTD_2D : {
       kernel = new polybench::POLYBENCH_FDTD_2D(run_params);
       break;
    }

//
// Apps kernels...
//
    case Apps_ENERGY : {
       kernel = new apps::ENERGY(run_params);
       break;
    }
    case Apps_PRESSURE : {
       kernel = new apps::PRESSURE(run_params);
       break;
    }
    case Apps_LULESH : {
       kernel = new apps::LULESH(run_params);
       break;
    }

    default: {
      std::cout << "\n Unknown Kernel ID = " << kid << std::endl;
    }

  } // end switch on kernel id

  return kernel;
}

}  // closing brace for rajaperf namespace
