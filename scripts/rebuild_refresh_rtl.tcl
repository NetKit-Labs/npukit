# Refresh BD module reference from rtl/, then full synth/impl/bit.
# Fixes stale packaged pl_logic after editing npukit_axil.sv etc.

set project_root [file normalize [file join [file dirname [info script]] ..]]
set project_file [file join $project_root vivado npukit.xpr]
set output_dir   [file join $project_root output]

catch {config_webtalk -user off}
catch {config_webtalk -install off}

open_project $project_file

# Make sure sources point at live rtl/
update_compile_order -fileset sources_1

open_bd_design [get_files system.bd]
# Refresh packaged SystemVerilog for the module-ref cell
if {[llength [get_bd_cells -quiet pl_logic]]} {
    update_module_reference [get_bd_cells pl_logic]
    puts "Updated module reference: pl_logic"
} else {
    error "BD cell pl_logic not found"
}
validate_bd_design
save_bd_design
close_bd_design [current_bd_design]

# Force bitstream path to rebuild
reset_run synth_1
reset_run impl_1

launch_runs synth_1 -jobs 2
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    error "Synthesis failed"
}
puts "Synthesis complete"

launch_runs impl_1 -to_step write_bitstream -jobs 2
wait_on_run impl_1

set bit_src [file join [get_property DIRECTORY [get_runs impl_1]] system_wrapper.bit]
if {![file exists $bit_src]} {
    error "Bitstream not generated: $bit_src"
}
file mkdir $output_dir
file copy -force $bit_src [file join $output_dir npukit.bit]
set hwh_files [get_files -quiet *system.hwh]
if {[llength $hwh_files] > 0} {
    file copy -force [lindex $hwh_files 0] [file join $output_dir npukit.hwh]
}
puts "Bitstream ready: [file join $output_dir npukit.bit]"
close_project
