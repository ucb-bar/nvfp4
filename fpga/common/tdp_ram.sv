// True dual-port block RAM on one clock. Each port reads (one cycle of latency, old data on a
// write) or writes. Writes of both ports to one address in one cycle leave it undefined.
module tdp_ram #(
    parameter int DEPTH = 1024,
    parameter int W     = 32
) (
    input  logic                     clk,
    input  logic                     a_en,
    input  logic                     a_we,
    input  logic [$clog2(DEPTH)-1:0] a_addr,
    input  logic [W-1:0]             a_wdata,
    output logic [W-1:0]             a_rdata,
    input  logic                     b_en,
    input  logic                     b_we,
    input  logic [$clog2(DEPTH)-1:0] b_addr,
    input  logic [W-1:0]             b_wdata,
    output logic [W-1:0]             b_rdata
);
    (* ram_style = "block" *) logic [W-1:0] mem [DEPTH];

    // The Vivado true dual-port template: one process per port.
    /* verilator lint_off MULTIDRIVEN */
    always @(posedge clk) begin
        if (a_en) begin
            if (a_we) mem[a_addr] <= a_wdata;
            a_rdata <= mem[a_addr];
        end
    end

    always @(posedge clk) begin
        if (b_en) begin
            if (b_we) mem[b_addr] <= b_wdata;
            b_rdata <= mem[b_addr];
        end
    end
    /* verilator lint_on MULTIDRIVEN */
endmodule
