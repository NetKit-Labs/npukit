# Force full synth+impl+bitstream after RTL changes.
# Usage: vivado -mode batch -source scripts/rebuild_force.tcl

set project_root [file normalize [file join [file dirname [info script]] ..]]
set project_file [file join $project_root vivado npukit.xpr]
set output_dir   [file join $project_root output]

catch {config_webtalk -user off}
catch {config_webtalk -install off}

open_project $project_file
# OOC module-ref must be reset or stale pl_logic.dcp is reused.
foreach r [get_runs -quiet *_synth_1] {
    if {[get_property PROGRESS $r] != "0%"} {
        catch {reset_run $r}
    }
}
reset_run synth_1
catch {reset_run impl_1}

launch_runs synth_1 -jobs 2
wait_on_run synth_1
set synth_progress [get_property PROGRESS [get_runs synth_1]]
if {$synth_progress != "100%"} {
    error "Synthesis failed (progress=$synth_progress)"
}
puts "Synthesis complete"

launch_runs impl_1 -to_step write_bitstream -jobs 2
wait_on_run impl_1

set impl_dir [get_property DIRECTORY [get_runs impl_1]]
set bit_src [file join $impl_dir system_wrapper.bit]
if {![file exists $bit_src]} {
    error "Bitstream not generated: $bit_src"
}

file mkdir $output_dir
set bit_out [file join $output_dir npukit.bit]
file copy -force $bit_src $bit_out
set hwh_files [get_files -quiet *system.hwh]
if {[llength $hwh_files] > 0} {
    file copy -force [lindex $hwh_files 0] [file join $output_dir npukit.hwh]
}

puts "Bitstream ready: $bit_out"
close_project
