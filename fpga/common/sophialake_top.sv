// SophiaLake (xc7a100tfgg484-1) top: 50 MHz board clock -> MMCM -> CLK_MHZ fabric clock.
// The host talks to the harness over the board's own USB UART (the Chipyard UART-TSI pins).
//
// led[0] heartbeat, led[1] toggles per result, led[2] harness error flag, led[3] MMCM locked.
module sophialake_top
    import dut_cfg_pkg::*;
#(
    parameter int CLK_MHZ = 50,
    parameter int BAUD    = 1_000_000
) (
    input  logic       sys_clock,
    input  logic       uart_tsi_rxd,
    output logic       uart_tsi_txd,
    output logic [3:0] led
);
    logic clk, rst, locked;

    board_clk_rst #(.CLK_MHZ(CLK_MHZ)) u_clk (.sys_clock, .clk, .rst, .locked);

    logic                   start, done, activity;
    logic [IN_BYTES*8-1:0]  in_data;
    logic [OUT_BYTES*8-1:0] out_data;
    logic [1:0]             flags;

    uart_harness #(
        .CLK_HZ   (CLK_MHZ * 1_000_000),
        .BAUD     (BAUD),
        .IN_BYTES (IN_BYTES),
        .OUT_BYTES(OUT_BYTES),
        .DESIGN_ID(DESIGN_ID)
    ) u_harness (
        .clk, .rst,
        .rxd(uart_tsi_rxd), .txd(uart_tsi_txd),
        .start, .in_data, .done, .out_data, .flags, .activity
    );

    dut_wrap u_dut (.clk, .rst, .start, .in_data, .done, .out_data);

    logic [25:0] beat = '0;
    always_ff @(posedge clk) beat <= beat + 1'b1;

    assign led = {locked, |flags, activity, beat[25]};
endmodule
