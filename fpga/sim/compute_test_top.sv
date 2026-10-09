// Direct MMIO simulation, plus arithmetic probes for GPU differential tests.
module compute_test_top
  import tl_pkg::*;
#(parameter int PM = 2, PN = 4) (
  input logic clk, rst,
  input logic a_valid,
  output logic a_ready,
  input a_t a,
  output logic d_valid,
  input logic d_ready,
  output d_t d,
  input logic [31:0] probe_a, probe_b,
  input vector_pkg::vector_op_e probe_operation,
  input logic probe_valid,
  output logic probe_ready, probe_complete,
  output logic [31:0] probe_result,
  output logic [31:0] probe_add, probe_mul, probe_exp2, probe_recip, probe_max
);
  compute_fabric #(.PM(PM), .PN(PN)) u_fabric (
    .clk, .rst, .a_valid, .a_ready, .a, .d_valid, .d_ready, .d,
    .dropped(1'b0), .bad_cmd(1'b0), .denied());
  fp32_alu u_probe_alu (.clk,.rst,.in_valid(probe_valid),.in_ready(probe_ready),
    .operation(probe_operation),.x(probe_a),.y(probe_b),
    .out_valid(probe_complete),.out_ready(1'b1),.result(probe_result));
  assign probe_add = fp32_pkg::fp_add(probe_a, probe_b);
  assign probe_mul = fp32_pkg::fp_mul(probe_a, probe_b);
  assign probe_exp2 = fp32_pkg::fp_exp2(probe_a);
  assign probe_recip = fp32_pkg::fp_recip(probe_a);
  assign probe_max = fp32_pkg::fp_max(probe_a, probe_b);
endmodule
