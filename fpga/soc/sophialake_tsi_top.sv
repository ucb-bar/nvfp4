// SophiaLake top for nvfp4_tsi_soc. The host runs PyUARTSI on the board's own USB UART (the
// Chipyard UART-TSI pins).
//
// led[0] heartbeat, led[1] toggles per TSI command, led[2] link error or denied access,
// led[3] MMCM locked.
module sophialake_tsi_top #(
    parameter int CLK_MHZ = 50,
    parameter int BAUD    = 921_600
) (
    input  logic       sys_clock,
    input  logic       uart_tsi_rxd,
    output logic       uart_tsi_txd,
    output logic [3:0] led
);
    logic clk, rst, locked, activity, error;

    board_clk_rst #(.CLK_MHZ(CLK_MHZ)) u_clk (.sys_clock, .clk, .rst, .locked);

    nvfp4_tsi_soc #(.CLK_HZ(CLK_MHZ * 1_000_000), .BAUD(BAUD)) u_soc (
        .clk, .rst, .rxd(uart_tsi_rxd), .txd(uart_tsi_txd), .activity, .error);

    logic [25:0] beat = '0;
    always_ff @(posedge clk) beat <= beat + 1'b1;

    assign led = {locked, error, activity, beat[25]};
endmodule
