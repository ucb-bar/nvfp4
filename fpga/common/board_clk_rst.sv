// SophiaLake clocking: 50 MHz board clock -> MMCM (VCO 1000 MHz) -> CLK_MHZ fabric clock,
// with reset held for 256 cycles after the MMCM locks.
module board_clk_rst #(
    parameter int CLK_MHZ = 50
) (
    input  logic sys_clock,
    output logic clk,
    output logic rst,
    output logic locked
);
    localparam real CLKOUT0_DIV = 1000.0 / CLK_MHZ;

    (* keep = "true" *) logic clk_in_ibuf;
    logic clk_fb, clk_mmcm;

    IBUF u_ibuf (.I(sys_clock), .O(clk_in_ibuf));

    MMCME2_BASE #(
        .CLKIN1_PERIOD    (20.0),
        .DIVCLK_DIVIDE    (1),
        .CLKFBOUT_MULT_F  (20.0),
        .CLKOUT0_DIVIDE_F (CLKOUT0_DIV)
    ) u_mmcm (
        .CLKIN1   (clk_in_ibuf),
        .CLKFBIN  (clk_fb),
        .CLKFBOUT (clk_fb),
        .CLKOUT0  (clk_mmcm),
        .LOCKED   (locked),
        .RST      (1'b0),
        .PWRDWN   (1'b0),
        .CLKFBOUTB(), .CLKOUT0B(), .CLKOUT1(), .CLKOUT1B(), .CLKOUT2(), .CLKOUT2B(),
        .CLKOUT3(), .CLKOUT3B(), .CLKOUT4(), .CLKOUT5(), .CLKOUT6()
    );

    BUFG u_bufg (.I(clk_mmcm), .O(clk));

    logic [8:0] por = '0;
    always_ff @(posedge clk or negedge locked) begin
        if (!locked) por <= '0;
        else if (!por[8]) por <= por + 1'b1;
    end
    always_ff @(posedge clk) rst <= !por[8];
endmodule
