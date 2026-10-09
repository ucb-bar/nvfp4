# Create the Vivado project for one design and run it through write_bitstream.
#   vivado -mode batch -source fpga/scripts/build.tcl -tclargs <design> [clk_mhz] [jobs]
set design  [lindex $argv 0]
set clk_mhz [expr {[llength $argv] > 1 ? [lindex $argv 1] : 50}]
set jobs    [expr {[llength $argv] > 2 ? [lindex $argv 2] : 8}]

set root    [file normalize [file join [file dirname [info script]] ../..]]
set common  "$root/fpga/common"
set dsgn    "$root/fpga/designs/$design"
set name    "${design}_${clk_mhz}mhz"
set out     "$root/build/$name"
set part    xc7a100tfgg484-1

# sources.f: one repo-relative path per line, +incdir+<dir> and +define+<macro> allowed.
# +top+<module> names a design with its own top: sources.f then lists every file, and the
# UART test harness, dut_cfg_pkg.sv and dut_wrap.sv are not added.
set srcs {}
set incdirs {}
set defines {}
set top sophialake_top
set fh [open "$dsgn/sources.f"]
foreach line [split [read $fh] "\n"] {
    set line [string trim $line]
    if {$line eq "" || [string index $line 0] eq "#"} continue
    if {[string match "+incdir+*" $line]} {
        lappend incdirs "$root/[string range $line 8 end]"
    } elseif {[string match "+define+*" $line]} {
        lappend defines [string range $line 8 end]
    } elseif {[string match "+top+*" $line]} {
        set top [string range $line 5 end]
    } else {
        lappend srcs "$root/$line"
    }
}
close $fh

create_project -force $name $out -part $part
set_property target_language Verilog [current_project]

if {$top eq "sophialake_top"} {
    set files [concat \
        [list "$dsgn/dut_cfg_pkg.sv"] $srcs [list "$dsgn/dut_wrap.sv"] \
        [list "$common/uart_rx.sv" "$common/uart_tx.sv" "$common/uart_harness.sv" \
              "$common/board_clk_rst.sv" "$common/sophialake_top.sv"]]
} else {
    set files $srcs
}
add_files -norecurse -fileset sources_1 $files
set_property include_dirs $incdirs [get_filesets sources_1]
set_property verilog_define $defines [get_filesets sources_1]
set_property top $top [get_filesets sources_1]
set_property generic "CLK_MHZ=$clk_mhz" [get_filesets sources_1]

add_files -norecurse -fileset constrs_1 "$common/sophialake.xdc"

launch_runs synth_1 -jobs $jobs
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} { error "synth_1 failed" }

launch_runs impl_1 -to_step write_bitstream -jobs $jobs
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} { error "impl_1 failed" }

open_run impl_1
report_utilization -file "$out/utilization.rpt"
report_utilization -hierarchical -hierarchical_depth 3 -file "$out/utilization_hier.rpt"
report_timing_summary -max_paths 10 -file "$out/timing_summary.rpt"
file copy -force [get_property DIRECTORY [get_runs impl_1]]/$top.bit "$out/$name.bit"

set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
puts "RESULT design=$name wns_ns=$wns bit=$out/$name.bit"
