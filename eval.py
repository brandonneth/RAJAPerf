#Running this script from a build directory will run evaluation
import os
import subprocess


def stencil_eval():
	num_passes = 10
	size = 'extralarge'
	kernels = "JACOBI_1D JACOBI_2D HEAT_3D HYDRO_2D FDTD_2D"
	prefix = 'Stencils'

	passes_option = '--npasses ' + str(num_passes)
	size_option = '--sizespec ' + size + ' --sizefact 10'
	outfile_option = '--outfile ' + prefix
	kernels_option = '--kernels ' + kernels

	command_string = " ".join(['./bin/raja-perf.exe', passes_option, size_option, outfile_option, kernels_option])

	return command_string

def pointwise_eval():
	num_passes = 10
	size = '10'
	kernels = "ENERGY PRESSURE GEN_LIN_RECUR"
	prefix = 'Pointwise'

	passes_option = '--npasses ' + str(num_passes)
	size_option = '--sizefact ' + size
	outfile_option = '--outfile ' + prefix
	kernels_option = '--kernels ' + kernels

	command_string = " ".join(['./bin/raja-perf.exe', passes_option, size_option, outfile_option, kernels_option])

	return command_string

print(stencil_eval())
print(pointwise_eval())

