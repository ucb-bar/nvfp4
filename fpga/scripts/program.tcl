# Program the SophiaLake FPGA over JTAG (volatile; the flash image returns on power cycle).
#   vivado -mode batch -source fpga/scripts/program.tcl -tclargs <bitfile>
set bitfile [file normalize [lindex $argv 0]]
if {![file exists $bitfile]} {
    puts "ERROR: $bitfile does not exist. Build it with: pixi run build <design> <mhz>"
    exit 1
}

# Other boards may share the hw_server (a VCU118 does): use only the SophiaLake JTAG target
# (Xilinx serial Q4FSQ41IV6A by default, or SOPHIALAKE_JTAG), and only if it holds an xc7a100t.
set serial [expr {[info exists ::env(SOPHIALAKE_JTAG)] ? $::env(SOPHIALAKE_JTAG) : "Q4FSQ41IV6A"}]
open_hw_manager
connect_hw_server -url localhost:3121
set targets [get_hw_targets -quiet "*/$serial"]
if {[llength $targets] != 1} {
    puts "ERROR: [llength $targets] JTAG targets match */$serial. Targets: [get_hw_targets]"
    exit 1
}
open_hw_target $targets
set dev [get_hw_devices -quiet xc7a100t*]
if {[llength $dev] != 1} {
    puts "ERROR: target $targets holds [get_hw_devices], not one xc7a100t"
    close_hw_target
    exit 1
}
puts "TARGET $targets device $dev"
current_hw_device $dev
set_property PROGRAM.FILE $bitfile $dev
program_hw_devices $dev
refresh_hw_device $dev
puts "RESULT done=[get_property REGISTER.CONFIG_STATUS.BIT14_DONE_PIN $dev] bit=$bitfile"
close_hw_target
