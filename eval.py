#Running this script from a build directory will run evaluation
import os
import subprocess


#returns the string of the output file name,
# made so its easy to parse later
def outfile_name(sizefact, repfact, passes):
	noFileExt = "_".join(("Results", str(sizefact), str(repfact), str(passes)))
	return noFileExt
#returns the command that, run from the build directory, 
#runs the suite with the arguments provided
def command_string(sizefact, repfact, passes=1):
	executable = "./bin/raja-perf.exe"
	size_option = "--sizefact " + str(sizefact)
	rep_option = "--repfact " + str(repfact)
	pass_option = "--npasses " + str(passes)
	outfile_option = "--outfile " + outfile_name(sizefact,repfact,passes)
	return " ".join((executable, size_option, rep_option, pass_option, outfile_option))




