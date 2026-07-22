# Create a PYNQ-Z2 Vivado project (Zynq PS board preset + NpuKit PL).
# Usage:
#   vivado -mode batch -source scripts/create_project.tcl

set project_root [file normalize [file join [file dirname [info script]] ..]]
set fpga_scripts [file normalize [file join $project_root .. scripts]]
set rtl_dir      [file join $project_root rtl]
set xdc_dir      [file join $project_root constraints]

set project_name "npukit"
set top_module   "npukit_pl"
set rtl_files [list \
    [file join $rtl_dir pe.sv] \
    [file join $rtl_dir systolic_array.sv] \
    [file join $rtl_dir npukit_top.sv] \
    [file join $rtl_dir npukit_pl.v] \
]
set xdc_file [file join $xdc_dir pynq_z2.xdc]

source [file join $fpga_scripts pynq_bitstream.tcl]
pynq_bitstream::create_project
