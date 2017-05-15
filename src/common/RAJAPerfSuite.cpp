/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file containing names of suite kernels and 
 *          variants, and routine for creating kernel objects.
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


#include "RAJAPerfSuite.hpp"

#include "RunParams.hpp"

//
// Basic kernels...
//
#include "basic/MULADDSUB.hpp"
#include "basic/IF_QUAD.hpp"
#include "basic/TRAP_INT.hpp"

//
// Livloops kernels...
//

//
// Polybench kernels...
//

//
// Stream kernels...
//
#include "stream/COPY.hpp"
#include "stream/MUL.hpp"
#include "stream/ADD.hpp"
#include "stream/TRIAD.hpp"
#include "stream/DOT.hpp"

//
// Apps kernels...
//
#include "apps/PRESSURE_CALC.hpp"
#include "apps/ENERGY_CALC.hpp"
#include "apps/VOL3D_CALC.hpp"
#include "apps/DEL_DOT_VEC_2D.hpp"
#include "apps/COUPLE.hpp"


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
  std::string("Basic"),
  std::string("Livloops"),
  std::string("Polybench"),
  std::string("Stream"),
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
// Basic kernels...
//
  std::string("Basic_MULADDSUB"),
  std::string("Basic_IF_QUAD"),
  std::string("Basic_TRAP_INT"),

//
// Livloops kernels...
//
#if 0
  std::string("Livloops_HYDRO_1D"),
  std::string("Livloops_ICCG"),
  std::string("Livloops_INNER_PROD"),
  std::string("Livloops_BAND_LIN_EQ"),
  std::string("Livloops_TRIDIAG_ELIM"),
  std::string("Livloops_EOS"),
  std::string("Livloops_ADI"),
  std::string("Livloops_INT_PREDICT"),
  std::string("Livloops_DIFF_PREDICT"),
  std::string("Livloops_FIRST_SUM"),
  std::string("Livloops_FIRST_DIFF"),
  std::string("Livloops_PIC_2D"),
  std::string("Livloops_PIC_1D"),
  std::string("Livloops_HYDRO_2D"),
  std::string("Livloops_GEN_LIN_RECUR"),
  std::string("Livloops_DISC_ORD"),
  std::string("Livloops_MAT_X_MAT"),
  std::string("Livloops_PLANCKIAN"),
  std::string("Livloops_IMP_HYDRO_2D"),
  std::string("Livloops_FIND_FIRST_MIN"),
#endif

//
// Polybench kernels...
//
#if 0
  std::string("Polybench_***");
#endif

//
// Stream kernels...
//
  std::string("Stream_COPY"),
  std::string("Stream_MUL"),
  std::string("Stream_ADD"),
  std::string("Stream_TRIAD"),
  std::string("Stream_DOT"),

//
// Apps kernels...
//
  std::string("Apps_PRESSURE_CALC"),
  std::string("Apps_ENERGY_CALC"),
  std::string("Apps_VOL3D_CALC"),
  std::string("Apps_DEL_DOT_VEC_2D"),
  std::string("Apps_COUPLE"),
#if 0
  std::string("Apps_FIR"),
#endif

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

  std::string("Baseline_Seq"),
  std::string("RAJA_Seq"),
#if defined(_OPENMP)
  std::string("Baseline_OpenMP"),
  std::string("RAJALike_OpenMP"),
  std::string("RAJA_OpenMP"),
#endif
#if defined(RAJA_ENABLE_CUDA)
  std::string("Baseline_CUDA"),
  std::string("RAJA_CUDA"),
#endif
#if 0
  std::string("Baseline_OpenMP4.x"),
  std::string("RAJA_OpenMP4.x"),
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
    // Basic kernels...
    //
    case Basic_MULADDSUB : {
       kernel = new basic::MULADDSUB(run_params);
       break;
    }
    case Basic_IF_QUAD : {
       kernel = new basic::IF_QUAD(run_params);
       break;
    }
    case Basic_TRAP_INT : {
       kernel = new basic::TRAP_INT(run_params);
       break;
    }

//
// Livloops kernels...
//
#if 0
  Livloops_HYDRO_1D,
  Livloops_ICCG,
  Livloops_INNER_PROD,
  Livloops_BAND_LIN_EQ,
  Livloops_TRIDIAG_ELIM,
  Livloops_EOS,
  Livloops_ADI,
  Livloops_INT_PREDICT,
  Livloops_DIFF_PREDICT,
  Livloops_FIRST_SUM,
  Livloops_FIRST_DIFF,
  Livloops_PIC_2D,
  Livloops_PIC_1D,
  Livloops_HYDRO_2D,
  Livloops_GEN_LIN_RECUR,
  Livloops_DISC_ORD,
  Livloops_MAT_X_MAT,
  Livloops_PLANCKIAN,
  Livloops_IMP_HYDRO_2D,
  Livloops_FIND_FIRST_MIN,
#endif

//
// Polybench kernels...
//
#if 0
  Polybench_***
#endif

//
// Stream kernels...
//
    case Stream_COPY : {
       kernel = new stream::COPY(run_params);
       break;
    }
    case Stream_MUL : {
       kernel = new stream::MUL(run_params);
       break;
    }
    case Stream_ADD : {
       kernel = new stream::ADD(run_params);
       break;
    }
    case Stream_TRIAD : {
       kernel = new stream::TRIAD(run_params);
       break;
    }
    case Stream_DOT : {
       kernel = new stream::DOT(run_params);
       break;
    }

//
// Apps kernels...
//
    case Apps_PRESSURE_CALC : {
       kernel = new apps::PRESSURE_CALC(run_params);
       break;
    }
    case Apps_ENERGY_CALC : {
       kernel = new apps::ENERGY_CALC(run_params);
       break;
    }
    case Apps_VOL3D_CALC : {
       kernel = new apps::VOL3D_CALC(run_params);
       break;
    }
    case Apps_DEL_DOT_VEC_2D : {
       kernel = new apps::DEL_DOT_VEC_2D(run_params);
       break;
    }
    case Apps_COUPLE : {
       kernel = new apps::COUPLE(run_params);
       break;
    }
#if 0
    case Apps_FIR : {
       kernel = new apps::FIR(run_params);
       break;
    }
#endif

    default: {
      std::cout << "\n Unknown Kernel ID = " << kid << std::endl;
    }

  } // end switch on kernel id

  return kernel;
}

}  // closing brace for rajaperf namespace