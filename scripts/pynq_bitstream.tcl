# Shared PYNQ-Z2 bitstream helpers (Zynq PS + PL → .bit).
# The PS block uses the board preset so Ethernet/DDR stay correct under Linux.
#
# Expected globals (set by the project script before sourcing this file):
#   project_name  - e.g. "blinker"  (Vivado project / output basename)
#   project_root  - project directory containing rtl/, constraints/, scripts/
#   rtl_files     - list of absolute RTL paths (create only)
#   top_module    - PL top module name (create only)
#   xdc_file      - absolute path to constraints (create only)
#   use_axi_lite  - optional; 1 = enable PS M_AXI_GP0 + FCLK0 to PL S_AXI (default 0)
#   use_axi_dma   - optional; 1 = add AXI DMA + PS HP0 for PL AXIS (default 0)
#
# Usage from a project:
#   set project_name blinker
#   ... set other vars ...
#   source [file join $fpga_scripts pynq_bitstream.tcl]
#   pynq_bitstream::create_project
#   pynq_bitstream::build

namespace eval pynq_bitstream {
    proc _use_axi {} {
        if {[info exists ::use_axi_lite] && $::use_axi_lite} {
            return 1
        }
        return 0
    }

    proc _use_dma {} {
        if {[info exists ::use_axi_dma] && $::use_axi_dma} {
            return 1
        }
        return 0
    }

    proc _webtalk_off {} {
        catch {config_webtalk -user off}
        catch {config_webtalk -install off}
    }

    proc _project_dir {} {
        return [file normalize [file join $::project_root vivado]]
    }

    proc _output_dir {} {
        return [file normalize [file join $::project_root output]]
    }

    proc _project_file {} {
        return [file join [_project_dir] ${::project_name}.xpr]
    }

    # Create / overwrite a Vivado project with Zynq PS + PL module reference.
    proc create_project {} {
        _webtalk_off

        set project_dir [_project_dir]
        file mkdir $project_dir
        ::create_project $::project_name $project_dir -part xc7z020clg400-1 -force
        set_property board_part tul.com.tw:pynq-z2:part0:1.0 [current_project]
        set_property target_language Verilog [current_project]
        set_property default_lib work [current_project]
        set_property source_mgmt_mode All [current_project]

        add_files -norecurse $::rtl_files
        foreach f $::rtl_files {
            set ext [string tolower [file extension $f]]
            if {$ext eq ".sv"} {
                set_property file_type SystemVerilog [get_files $f]
            } elseif {$ext eq ".v"} {
                set_property file_type Verilog [get_files $f]
            }
        }
        update_compile_order -fileset sources_1

        create_bd_design "system"

        set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0]
        apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {
            make_external {FIXED_IO DDR}
            apply_board_preset 1
            Master Disable
            Slave Disable
        } $ps

        create_bd_cell -type module -reference $::top_module pl_logic

        create_bd_port -dir I btn0
        create_bd_port -dir O -from 3 -to 0 led
        connect_bd_net [get_bd_ports btn0] [get_bd_pins pl_logic/btn0]
        connect_bd_net [get_bd_ports led]  [get_bd_pins pl_logic/led]

        if {[_use_axi]} {
            # PS master GP0 → AXI interconnect → PL S_AXI; PL clock = FCLK0 (100 MHz)
            set_property -dict [list \
                CONFIG.PCW_USE_M_AXI_GP0 {1} \
                CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} \
            ] $ps

            apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list \
                Master "/processing_system7_0/M_AXI_GP0" \
                Clk    "Auto" \
            ] [get_bd_intf_pins pl_logic/S_AXI]

            # Stable MMIO base for Bitstream+MMIO host (no .hwh required)
            assign_bd_address -force
            set host_segs [get_bd_addr_segs -quiet -of_objects \
                [get_bd_addr_spaces processing_system7_0/Data]]
            set mapped 0
            foreach s $host_segs {
                if {[string match -nocase "*pl_logic*" $s]} {
                    set_property offset 0x43C00000 $s
                    set_property range  64K        $s
                    set mapped 1
                    puts "AXI-Lite map: $s → 0x43C00000 / 64K"
                }
            }
            if {!$mapped} {
                # Fallback: any segment that includes pl_logic in the slave path
                foreach s [get_bd_addr_segs -quiet] {
                    if {[string match -nocase "*pl_logic*" $s]} {
                        catch {
                            set_property offset 0x43C00000 $s
                            set_property range  64K        $s
                            puts "AXI-Lite map (fallback): $s → 0x43C00000 / 64K"
                            set mapped 1
                        }
                    }
                }
            }
            if {!$mapped} {
                error "Failed to map pl_logic S_AXI to 0x43C00000"
            }
            puts "AXI-Lite enabled: PS M_AXI_GP0 → $::top_module/S_AXI @ 0x43C00000"

            if {[_use_dma]} {
                # Simple-mode DMA: MM2S supplies the A/B tile and S2MM receives
                # the C tile.  Both data movers share the PS DDR HP0 port.
                set_property CONFIG.PCW_USE_S_AXI_HP0 {1} $ps
                create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0
                set_property -dict [list \
                    CONFIG.c_include_sg {0} \
                    CONFIG.c_include_mm2s_dre {0} \
                    CONFIG.c_include_s2mm_dre {0} \
                    CONFIG.c_sg_include_stscntrl_strm {0} \
                    CONFIG.c_m_axi_mm2s_data_width {32} \
                    CONFIG.c_m_axi_s2mm_data_width {32} \
                    CONFIG.c_m_axis_mm2s_tdata_width {32} \
                    CONFIG.c_s_axis_s2mm_tdata_width {32} \
                ] [get_bd_cells axi_dma_0]

                connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] \
                    [get_bd_intf_pins pl_logic/S_AXIS]
                connect_bd_intf_net [get_bd_intf_pins pl_logic/M_AXIS] \
                    [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]

                create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 hp0_interconnect
                set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] \
                    [get_bd_cells hp0_interconnect]
                connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] \
                    [get_bd_intf_pins hp0_interconnect/S00_AXI]
                connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] \
                    [get_bd_intf_pins hp0_interconnect/S01_AXI]
                connect_bd_intf_net [get_bd_intf_pins hp0_interconnect/M00_AXI] \
                    [get_bd_intf_pins processing_system7_0/S_AXI_HP0]

                set fclk [get_bd_pins processing_system7_0/FCLK_CLK0]
                set frst [get_bd_pins processing_system7_0/FCLK_RESET0_N]
                foreach pin [list \
                    axi_dma_0/s_axi_lite_aclk axi_dma_0/m_axi_mm2s_aclk \
                    axi_dma_0/m_axi_s2mm_aclk hp0_interconnect/ACLK \
                    hp0_interconnect/S00_ACLK hp0_interconnect/S01_ACLK \
                    hp0_interconnect/M00_ACLK processing_system7_0/S_AXI_HP0_ACLK] {
                    connect_bd_net $fclk [get_bd_pins $pin]
                }
                # Temporary async reset; rewired to proc_sys_reset after GP0 automation.
                connect_bd_net $frst [get_bd_pins axi_dma_0/axi_resetn]
                connect_bd_net $frst [get_bd_pins hp0_interconnect/ARESETN]
                connect_bd_net $frst [get_bd_pins hp0_interconnect/S00_ARESETN]
                connect_bd_net $frst [get_bd_pins hp0_interconnect/S01_ARESETN]
                connect_bd_net $frst [get_bd_pins hp0_interconnect/M00_ARESETN]

                apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list \
                    Master "/processing_system7_0/M_AXI_GP0" \
                    Clk    "Auto" \
                ] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

                if {[llength [get_bd_cells -quiet rst_ps7_0_100M]] > 0} {
                    set peri_rstn [get_bd_pins rst_ps7_0_100M/peripheral_aresetn]
                    set ic_rstn   [get_bd_pins rst_ps7_0_100M/interconnect_aresetn]
                    foreach pin [list \
                        axi_dma_0/axi_resetn \
                        hp0_interconnect/S00_ARESETN \
                        hp0_interconnect/S01_ARESETN \
                        hp0_interconnect/M00_ARESETN] {
                        catch {disconnect_bd_net [get_bd_nets -quiet -of_objects [get_bd_pins $pin]] \
                            [get_bd_pins $pin]}
                        connect_bd_net $peri_rstn [get_bd_pins $pin]
                    }
                    catch {disconnect_bd_net \
                        [get_bd_nets -quiet -of_objects [get_bd_pins hp0_interconnect/ARESETN]] \
                        [get_bd_pins hp0_interconnect/ARESETN]}
                    connect_bd_net $ic_rstn [get_bd_pins hp0_interconnect/ARESETN]
                }

                assign_bd_address -force
                foreach s [get_bd_addr_segs -quiet -of_objects \
                    [get_bd_addr_spaces processing_system7_0/Data]] {
                    if {[string match -nocase "*axi_dma_0*" $s]} {
                        set_property offset 0x40400000 $s
                        set_property range  64K        $s
                        puts "AXI DMA map: $s → 0x40400000 / 64K"
                    }
                }
                puts "AXI DMA enabled: MM2S/S2MM → HP0"
            }
        } else {
            # No AXI — external 125 MHz board clock (blinker-style)
            set_property -dict [list CONFIG.PCW_USE_M_AXI_GP0 {0}] $ps

            create_bd_port -dir I -type clk -freq_hz 125000000 clk
            connect_bd_net [get_bd_ports clk] [get_bd_pins pl_logic/clk]
            set_property CONFIG.FREQ_HZ 125000000 [get_bd_ports clk]
        }

        regenerate_bd_layout
        save_bd_design
        validate_bd_design
        save_bd_design

        set bd_file [get_files system.bd]
        make_wrapper -files $bd_file -top
        set wrapper_files [get_files -quiet *_wrapper.v]
        if {[llength $wrapper_files] == 0} {
            set wrapper_files [glob -nocomplain \
                [file join $project_dir ${::project_name}.gen sources_1 bd system hdl *_wrapper.v] \
                [file join $project_dir ${::project_name}.srcs sources_1 bd system hdl *_wrapper.v]]
            if {[llength $wrapper_files] == 0} {
                error "system_wrapper.v not found after make_wrapper"
            }
            add_files -norecurse $wrapper_files
        }
        update_compile_order -fileset sources_1
        set_property top system_wrapper [current_fileset]

        add_files -fileset constrs_1 -norecurse $::xdc_file
        update_compile_order -fileset sources_1

        puts "Created PYNQ-Z2 project '$::project_name'"
        puts "  Top: system_wrapper (PS + $::top_module)"
        puts "  Dir: $project_dir"
        close_project
    }

    # Build bitstream: output/<name>.bit
    proc build {} {
        _webtalk_off

        set project_dir  [_project_dir]
        set project_file [_project_file]
        set output_dir   [_output_dir]

        if {![file exists $project_file]} {
            error "Project not found: $project_file — run create first"
        }

        open_project $project_file

        set synth_progress [get_property PROGRESS [get_runs synth_1]]
        if {$synth_progress != "100%"} {
            reset_run synth_1
            launch_runs synth_1 -jobs 2
            wait_on_run synth_1
            set synth_progress [get_property PROGRESS [get_runs synth_1]]
        }
        if {$synth_progress != "100%"} {
            error "Synthesis failed"
        }
        puts "Synthesis complete"

        set impl_dir [get_property DIRECTORY [get_runs impl_1]]
        set bit_src [file join $impl_dir system_wrapper.bit]

        if {![file exists $bit_src]} {
            if {[get_property STATUS [get_runs impl_1]] != "Not started"} {
                reset_run impl_1
            }
            launch_runs impl_1 -to_step write_bitstream -jobs 2
            wait_on_run impl_1
        }

        if {![file exists $bit_src]} {
            error "Bitstream not generated: $bit_src"
        }

        file mkdir $output_dir
        set bit_out [file join $output_dir ${::project_name}.bit]
        file copy -force $bit_src $bit_out
        set hwh_files [get_files -quiet *system.hwh]
        if {[llength $hwh_files] > 0} {
            file copy -force [lindex $hwh_files 0] \
                [file join $output_dir ${::project_name}.hwh]
        } else {
            puts "WARNING: system.hwh not found; Overlay metadata not copied"
        }

        puts "Bitstream ready:"
        puts "  $bit_out"
        puts "Load on PYNQ with:"
        puts "  from pynq import Bitstream"
        puts "  Bitstream(\"/path/${::project_name}.bit\").download()"
        close_project
    }
}
