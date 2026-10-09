# Reproducible area and post-synthesis timing estimate for the compute SoC.
set root [file normalize [file join [file dirname [info script]] ../..]]
set out "$root/build/compute_synth"
file mkdir $out
set files {}
set includes {}
set fh [open "$root/fpga/designs/nvfp4_tsi/sources.f"]
foreach line [split [read $fh] "\n"] {
    set line [string trim $line]
    if {$line eq "" || [string index $line 0] eq "#" || [string match "+top+*" $line]} continue
    if {[string match "+incdir+*" $line]} {
        lappend includes "$root/[string range $line 8 end]"
    } elseif {![string match "*board_clk_rst*" $line] && ![string match "*sophialake_tsi_top*" $line]} {
        lappend files "$root/$line"
    }
}
close $fh
set_part xc7a100tfgg484-1
set_property include_dirs $includes [current_fileset]
read_verilog -sv $files
synth_design -top nvfp4_tsi_soc -part xc7a100tfgg484-1
create_clock -period 20 [get_ports clk]
report_utilization -file "$out/utilization.rpt"
report_utilization -hierarchical -file "$out/utilization_hier.rpt"
report_timing_summary -max_paths 5 -file "$out/timing_summary.rpt"
write_checkpoint -force "$out/compute.dcp"
