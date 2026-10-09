// UART TSI target -> compute_fabric. Address map:
//
//   0x0000_0000 - 0x0000_0FFF  sys_regs        link identification and status
//   0x1000_0000 - 0x1000_FFFF  nvfp4_tc        NVFP4 tensor core and its tile memories
//   0x1001_0000 - 0x1001_FFFF  flash_vector    eight-lane FP32 vector unit, NVFP4 quantization
//   0x1002_0000 - 0x1003_FFFF  kernel_engine   programmable sequencer, second bus master
//   0x2000_0000 - 0x2001_FFFF  scratch_memory  128 KiB
//
// Unmapped accesses get a denied response.
module nvfp4_tsi_soc
    import tl_pkg::*;
#(
    parameter int CLK_HZ = 50_000_000,
    parameter int BAUD   = 921_600
) (
    input  logic clk,
    input  logic rst,
    input  logic rxd,
    output logic txd,
    output logic activity,  // toggles when a TSI command completes
    output logic error      // a link error or denied access has occurred
);
    localparam int CLKS_PER_BIT = (CLK_HZ + BAUD / 2) / BAUD;

    logic a_valid, a_ready, d_valid, d_ready;
    a_t   a;
    d_t   d;
    logic dropped, bad_cmd, done, denied;

    uart_tsi #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_tsi (
        .clk, .rst, .rxd, .txd,
        .a_valid, .a_ready, .a, .d_valid, .d_ready, .d,
        .dropped, .bad_cmd, .done);

    compute_fabric #(.CLK_HZ(CLK_HZ), .BAUD(BAUD)) u_fabric (
        .clk, .rst, .a_valid, .a_ready, .a, .d_valid, .d_ready, .d,
        .dropped, .bad_cmd, .denied);

    logic denied_seen;
    always_ff @(posedge clk) begin
        if (rst) begin
            activity    <= 1'b0;
            denied_seen <= 1'b0;
        end else begin
            if (done) activity <= ~activity;
            if (denied) denied_seen <= 1'b1;
        end
    end
    assign error = dropped | bad_cmd | denied_seen;
endmodule
