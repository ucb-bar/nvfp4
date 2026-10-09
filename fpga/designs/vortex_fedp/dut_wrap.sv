`include "VX_define.vh"

// vortexgpgpu/vortex VX_tcu_fedp_tfr, parameterized the way VX_tcu_core instantiates it.
// Fixed-latency pipeline: d_val is valid VX_CFG_TCU_LATENCY cycles after the inputs.
module dut_wrap
    import dut_cfg_pkg::*;
    import VX_tcu_pkg::*;
(
    input  logic                   clk,
    input  logic                   rst,
    input  logic                   start,
    input  logic [IN_BYTES*8-1:0]  in_data,
    output logic                   done,
    output logic [OUT_BYTES*8-1:0] out_data
);
    localparam int N   = TCU_TC_K;
    localparam int SF  = TCU_MX_MAX_SF;
    localparam int LAT = `VX_CFG_TCU_LATENCY;

    if (IN_BYTES != 8 * N + 2 * SF + 5) $error("IN_BYTES does not match the FEDP geometry");

    logic [N-1:0][31:0]  a_row, b_col;
    logic [SF-1:0][7:0]  sf_a, sf_b;
    logic [31:0]         c_val;
    logic [4:0]          fmt_s;
    logic [LAT:0]        vld;

    always_ff @(posedge clk) begin
        if (start) {fmt_s, c_val, sf_b, sf_a, b_col, a_row} <= in_data[IN_BYTES*8-4:0];
        vld <= rst ? '0 : {vld[LAT-1:0], start};
    end

    VX_tcu_fedp_tfr #(
        .LATENCY (LAT),
        .N       (N),
        .SF      (SF),
        .USE_DSP (1)
    ) fedp (
        .clk, .reset(rst), .enable(1'b1),
        .vld_mask('1), .fmt_s, .fmt_d(5'(TCU_FP32_ID)),
        .a_row, .b_col, .sf_a, .sf_b, .c_val,
        .d_val(out_data)
    );

    assign done = vld[LAT];
endmodule
