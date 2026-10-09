# SophiaLake bring-up board, xc7a100tfgg484-1.
# Pins come from the Chipyard SophiaLake shell (fpga-shells/xilinx/sophialake).

# 50 MHz oscillator. W19 is not a clock-capable pin.
set_property -dict {PACKAGE_PIN W19 IOSTANDARD LVCMOS33} [get_ports sys_clock]
create_clock -name sys_clock -period 20.000 [get_ports sys_clock]
set_property CLOCK_DEDICATED_ROUTE FALSE [get_nets -hierarchical clk_in_ibuf]

# USB UART on the board's own "FPGA" USB-C port (Chipyard UART-TSI pins).
set_property -dict {PACKAGE_PIN T21 IOSTANDARD LVCMOS33 PULLTYPE PULLUP} [get_ports uart_tsi_rxd]
set_property -dict {PACKAGE_PIN U21 IOSTANDARD LVCMOS33} [get_ports uart_tsi_txd]

set_property -dict {PACKAGE_PIN V18 IOSTANDARD LVCMOS33} [get_ports {led[0]}]
set_property -dict {PACKAGE_PIN Y18 IOSTANDARD LVCMOS33} [get_ports {led[1]}]
set_property -dict {PACKAGE_PIN P19 IOSTANDARD LVCMOS33} [get_ports {led[2]}]
set_property -dict {PACKAGE_PIN U20 IOSTANDARD LVCMOS33} [get_ports {led[3]}]

# Asynchronous I/O: the UART input goes through a synchronizer, the outputs are slow.
set_false_path -from [get_ports uart_tsi_rxd]
set_false_path -to [get_ports {uart_tsi_txd led[*]}]

set_property CONFIG_VOLTAGE 3.3 [current_design]
set_property CFGBVS VCCO [current_design]
