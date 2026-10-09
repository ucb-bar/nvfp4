// fp4_fedp_dsp (fpga/pe/fp4_fedp_dsp.sv) with N = 2: rows (a, b) and columns (b, a) of one case,
// so d[0][0] = a.b is the case and the four packed product fields all carry data.
// Fixed 5-cycle latency, inputs registered on start.
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
    logic [255:0] a, b;
    logic [31:0]  sfa, sfb, c;
    logic [1:0]   mode;
    logic         vld;
    logic [31:0]  d [2][2];

    always_ff @(posedge clk) begin
        vld <= !rst && start;
        if (start) {mode, c, sfb, sfa, b, a} <= in_data[609:0];
    end

    fp4_fedp_dsp #(.N(2)) u_pe (
        .clk, .rst, .in_vld(vld), .a('{a, b}), .b('{b, a}), .mode,
        .sfa('{sfa, sfb}), .sfb('{sfb, sfa}), .c('{'{c, c}, '{c, c}}), .out_vld(done), .d);
    assign out_data = {d[1][1], d[1][0], d[0][1], d[0][0]};
endmodule
