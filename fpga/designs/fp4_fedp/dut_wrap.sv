// fp4_fedp (fpga/pe/fp4_fedp.sv): fixed 5-cycle latency, inputs registered on start.
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

    always_ff @(posedge clk) begin
        vld <= !rst && start;
        if (start) {mode, c, sfb, sfa, b, a} <= in_data[609:0];
    end

    fp4_fedp u_pe (.clk, .rst, .in_vld(vld), .a, .b, .mode, .sfa, .sfb, .c, .out_vld(done), .d(out_data));
endmodule
