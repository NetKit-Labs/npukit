# Create a PYNQ-Z2 Vivado project (Zynq PS + AXI-Lite NpuKit PL).
# Usage:
#   vivado -mode batch -source scripts/create_project.tcl

set project_root [file normalize [file join [file dirname [info script]] ..]]
set scripts_dir  [file join $project_root scripts]
set rtl_dir      [file join $project_root rtl]
set xdc_dir      [file join $project_root constraints]

set project_name "npukit"
set top_module   "npukit_pl"
set use_axi_lite 1
set use_axi_dma  1
set rtl_files [list \
    [file join $rtl_dir pe.sv] \
    [file join $rtl_dir systolic_array.sv] \
    [file join $rtl_dir npukit_glue.sv] \
    [file join $rtl_dir npukit_axil.sv] \
    [file join $rtl_dir npukit_top.sv] \
    [file join $rtl_dir npukit_pl.v] \
]
set xdc_file [file join $xdc_dir pynq_z2.xdc]

source [file join $scripts_dir pynq_bitstream.tcl]
pynq_bitstream::create_project
