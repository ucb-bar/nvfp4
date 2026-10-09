`timescale 1ns/1ps
// flash_vector_unit handshakes: operand capture, output backpressure, an all-masked SOFTMAX, reset.
module tb_vector;
  import vector_pkg::*;
  logic clk = 0, rst = 1, in_valid = 0, in_ready, out_valid, out_ready = 0;
  vector_op_e operation = V_ADD;
  logic [7:0] lane_mask = '1;
  logic [31:0] x [8], y [8], result [8];
  logic [31:0] scalar = fp32_pkg::ONE, old_max = fp32_pkg::NEG_INF, old_sum = 0;
  logic [31:0] reduction, new_max, new_sum, alpha;
  always #5 clk = ~clk;
  flash_vector_unit dut (.*);
  initial begin
    logic [31:0] held [8];
    for (int i = 0; i < 8; i++) begin x[i] = fp32_pkg::ONE; y[i] = fp32_pkg::ONE; end
    repeat (3) @(negedge clk);
    rst = 0;
    in_valid = 1;
    @(negedge clk);
    in_valid = 0;
    // Changes to input ports during a command must not alter captured operands.
    x[0] = 0;
    while (!out_valid) @(negedge clk);
    for (int i = 0; i < 8; i++) if (result[i] !== 32'h40000000) $fatal(1,"captured ADD operands");
    held = result;
    in_valid = 1;
    repeat (5) begin
      @(negedge clk);
      if (in_ready || !out_valid || result !== held) $fatal(1,"output backpressure");
    end
    in_valid = 0;
    out_ready = 1;
    @(negedge clk);
    operation = V_SOFTMAX;
    lane_mask = 0;
    in_valid = 1;
    @(negedge clk);
    in_valid = 0;
    out_ready = 0;
    while (!out_valid) @(negedge clk);
    if (new_max !== fp32_pkg::NEG_INF || new_sum !== 0 || alpha !== 0)
      $fatal(1,"empty initial softmax state");
    for (int i = 0; i < 8; i++) if (result[i] !== 0) $fatal(1,"masked probability");
    out_ready = 1;
    @(negedge clk);
    in_valid = 1;
    lane_mask = '1;
    @(negedge clk);
    in_valid = 0;
    repeat (3) @(negedge clk);
    rst = 1;
    @(negedge clk);
    rst = 0;
    repeat (30) begin
      @(negedge clk);
      if (out_valid || !in_ready) $fatal(1,"reset did not cancel command");
    end
    $display("PASS vector capture, backpressure, empty masks, reset");
    $finish;
  end
  initial begin #10000; $fatal(1,"timeout"); end
endmodule
