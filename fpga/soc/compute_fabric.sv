// The compute devices behind one 32-bit host port: UART TSI on the board, direct MMIO in simulation
// (compute_test_top). The host port carries 32-bit words; the device fabric carries 32-byte beats.
// Register devices take one word at a time (tl_narrow); scratch memory and the tensor core take
// beats. The vector unit streams its operands through a second scratch port.
module compute_fabric
  import tl_pkg::*;
#(parameter int PM = 2, PN = 4,
  parameter int CLK_HZ = 50_000_000, BAUD = 921_600) (
  input logic clk, rst,
  input logic a_valid,
  output logic a_ready,
  input a_t a,
  output logic d_valid,
  input logic d_ready,
  output d_t d,
  input logic dropped, bad_cmd,
  output logic denied
);
  localparam int NDEV = 5;
  localparam int BEATS = 4096;  // 128 KiB of scratch
  logic [1:0] host_a_valid, host_a_ready, host_d_valid, host_d_ready;
  wa_t host_a [2];
  wd_t host_d;
  logic bus_a_valid, bus_a_ready, bus_d_valid, bus_d_ready;
  wa_t bus_a;
  wd_t bus_d;
  logic [NDEV-1:0] dev_a_valid, dev_a_ready, dev_d_valid, dev_d_ready;
  wa_t dev_a;
  wd_t dev_d [NDEV];
  logic v_en, v_we;
  logic [7:0] v_wmask;
  logic [$clog2(BEATS)-1:0] v_addr;
  logic [255:0] v_wdata, v_rdata;
  tl_widen u_widen (
    .clk, .a_valid, .a_ready, .a, .d_valid, .d_ready, .d,
    .w_a_valid(host_a_valid[0]), .w_a_ready(host_a_ready[0]), .w_a(host_a[0]),
    .w_d_valid(host_d_valid[0]), .w_d_ready(host_d_ready[0]), .w_d(host_d));
  tl_arbiter u_arbiter (
    .clk, .rst, .host_a_valid, .host_a_ready, .host_a, .host_d_valid, .host_d_ready, .host_d,
    .a_valid(bus_a_valid), .a_ready(bus_a_ready), .a(bus_a),
    .d_valid(bus_d_valid), .d_ready(bus_d_ready), .d(bus_d));
  tl_xbar #(.N(NDEV),
    .BASE({32'h2000_0000, 32'h1002_0000, 32'h1001_0000, 32'h1000_0000, 32'h0000_0000}),
    .MASK({32'h0001_ffff, 32'h1ffff, 32'hffff, 32'hffff, 32'hfff})) u_xbar (
    .clk, .rst, .a_valid(bus_a_valid), .a_ready(bus_a_ready), .a(bus_a),
    .d_valid(bus_d_valid), .d_ready(bus_d_ready), .d(bus_d),
    .dev_a_valid, .dev_a_ready, .dev_a, .dev_d_valid, .dev_d_ready, .dev_d, .denied);

  // Devices: 0 sys_regs, 1 nvfp4_tc, 2 flash_vector, 3 kernel_engine, 4 scratch_memory.
  // The register devices 0, 2 and 3 sit behind tl_narrow.
  logic [NDEV-1:0] n_a_valid, n_a_ready, n_d_valid, n_d_ready;
  a_t n_a [NDEV];
  d_t n_d [NDEV];
  for (genvar k = 0; k < NDEV; k++) begin : g_narrow
    if (k == 0 || k == 2 || k == 3) begin : g_dev
      tl_narrow u_narrow (
        .clk, .rst, .a_valid(dev_a_valid[k]), .a_ready(dev_a_ready[k]), .a(dev_a),
        .d_valid(dev_d_valid[k]), .d_ready(dev_d_ready[k]), .d(dev_d[k]),
        .n_a_valid(n_a_valid[k]), .n_a_ready(n_a_ready[k]), .n_a(n_a[k]),
        .n_d_valid(n_d_valid[k]), .n_d_ready(n_d_ready[k]), .n_d(n_d[k]));
    end else begin : g_none
      assign n_a_valid[k] = 0;
      assign n_a_ready[k] = 0;
      assign n_a[k] = '0;
      assign n_d_valid[k] = 0;
      assign n_d_ready[k] = 0;
      assign n_d[k] = '0;
    end
  end

  sys_regs #(.CLK_HZ(CLK_HZ), .BAUD(BAUD)) u_sys (
    .clk, .rst, .a_valid(n_a_valid[0]), .a_ready(n_a_ready[0]), .a(n_a[0]),
    .d_valid(n_d_valid[0]), .d_ready(n_d_ready[0]), .d(n_d[0]), .dropped, .bad_cmd, .denied);
  nvfp4_tc #(.PM(PM), .PN(PN)) u_tc (
    .clk, .rst, .a_valid(dev_a_valid[1]), .a_ready(dev_a_ready[1]), .a(dev_a),
    .d_valid(dev_d_valid[1]), .d_ready(dev_d_ready[1]), .d(dev_d[1]), .busy());
  flash_vector #(.BEATS(BEATS)) u_vector (
    .clk, .rst, .a_valid(n_a_valid[2]), .a_ready(n_a_ready[2]), .a(n_a[2]),
    .d_valid(n_d_valid[2]), .d_ready(n_d_ready[2]), .d(n_d[2]),
    .v_en, .v_we, .v_wmask, .v_addr, .v_wdata, .v_rdata, .busy());
  kernel_engine u_kernel (
    .clk, .rst, .a_valid(n_a_valid[3]), .a_ready(n_a_ready[3]), .a(n_a[3]),
    .d_valid(n_d_valid[3]), .d_ready(n_d_ready[3]), .d(n_d[3]), .busy(),
    .master_a_valid(host_a_valid[1]), .master_a_ready(host_a_ready[1]), .master_a(host_a[1]),
    .master_d_valid(host_d_valid[1]), .master_d_ready(host_d_ready[1]), .master_d(host_d));
  scratch_memory #(.BEATS(BEATS)) u_scratch (
    .clk, .rst, .a_valid(dev_a_valid[4]), .a_ready(dev_a_ready[4]), .a(dev_a),
    .d_valid(dev_d_valid[4]), .d_ready(dev_d_ready[4]), .d(dev_d[4]),
    .v_en, .v_we, .v_wmask, .v_addr, .v_wdata, .v_rdata);
endmodule
