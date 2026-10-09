// Simple dual-port block RAM: one write port, one read port with one cycle of latency.
// rdata changes only on a cycle with re set.
module sdp_ram #(
    parameter int DEPTH = 512,
    parameter int W     = 32
) (
    input  logic                     clk,
    input  logic                     we,
    input  logic [$clog2(DEPTH)-1:0] waddr,
    input  logic [W-1:0]             wdata,
    input  logic                     re,
    input  logic [$clog2(DEPTH)-1:0] raddr,
    output logic [W-1:0]             rdata
);
    (* ram_style = "block" *) logic [W-1:0] mem [DEPTH];

    always_ff @(posedge clk) begin
        if (we) mem[waddr] <= wdata;
        if (re) rdata <= mem[raddr];
    end
endmodule
