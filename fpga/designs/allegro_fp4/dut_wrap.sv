// roundandr/Allegro: 64-element FP4 dot product (4 x NVFP4 block16) + FP32 accumulate.
module dut_wrap
    import dut_cfg_pkg::*;
(
    input  logic                   clk,
    input  logic                   rst,
    input  logic                   start,
    input  logic [IN_BYTES*8-1:0]  in_data,
    output logic                   done,
    output logic [OUT_BYTES*8-1:0] out_data
);
    logic [255:0] a_fp4, b_fp4;
    logic [31:0]  a_sf, b_sf, c_fp32;
    logic [1:0]   fp4_mode;
    logic         in_vld, in_rdy;

    always_ff @(posedge clk) begin
        if (rst) begin
            in_vld <= 1'b0;
        end else if (start) begin
            {fp4_mode, c_fp32, b_sf, a_sf, b_fp4, a_fp4} <= in_data[609:0];
            in_vld <= 1'b1;
        end else if (in_rdy) begin
            in_vld <= 1'b0;
        end
    end

    fp4_dot_prod u_fp4 (
        .clk, .rst_n(!rst),
        .in_vld_i(in_vld), .in_rdy_o(in_rdy),
        .a_fp4_i(a_fp4), .b_fp4_i(b_fp4), .fp4_mode_i(fp4_mode),
        .a_sf_i(a_sf), .b_sf_i(b_sf), .c_fp32_i(c_fp32),
        .out_vld_o(done), .out_rdy_i(1'b1), .d_fp32_o(out_data)
    );
endmodule
