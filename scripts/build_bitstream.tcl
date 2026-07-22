# Build bitstream: output/npukit.bit
# Usage:
#   vivado -mode batch -source scripts/build_bitstream.tcl
#
# On the board:
#   from pynq import Bitstream
#   Bitstream("/home/xilinx/jupyter_notebooks/npukit.bit").download()

set project_root [file normalize [file join [file dirname [info script]] ..]]
set fpga_scripts [file normalize [file join $project_root .. scripts]]

set project_name "npukit"
set project_root $project_root

source [file join $fpga_scripts pynq_bitstream.tcl]
pynq_bitstream::build
