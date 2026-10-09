// System registers: link identification, a scratch word, and link error status.
//
//   0x000  ID       RO  0x5453_4930 ("TSI0")
//   0x004  SCRATCH  RW
//   0x008  FLAGS    RO  [0] receive queue overflowed, [1] bad TSI cmd, [2] a denied access occurred
//   0x00C  DENIED   RO  number of denied (unmapped) accesses
//   0x010  CYCLES   RO  free-running cycle counter
//   0x014  CLK_HZ   RO
//   0x018  BAUD     RO
module sys_regs
    import tl_pkg::*;
#(
    parameter int CLK_HZ = 50_000_000,
    parameter int BAUD   = 921_600
) (
    input  logic clk,
    input  logic rst,
    input  logic a_valid,
    output logic a_ready,
    input  a_t   a,
    output logic d_valid,
    input  logic d_ready,
    output d_t   d,

    input  logic dropped,
    input  logic bad_cmd,
    input  logic denied
);
    logic [31:0] scratch, denied_count, cycles, rdata;
    logic        hit;

    wire [9:0] word = a.address[11:2];
    wire       wr   = a.opcode != A_GET;

    always_comb begin
        hit = 1'b1;
        case (word)
            10'h0:   rdata = 32'h5453_4930;
            10'h1:   rdata = scratch;
            10'h2:   rdata = {29'd0, denied_count != 0, bad_cmd, dropped};
            10'h3:   rdata = denied_count;
            10'h4:   rdata = cycles;
            10'h5:   rdata = CLK_HZ;
            10'h6:   rdata = BAUD;
            default: begin
                rdata = '0;
                hit   = 1'b0;
            end
        endcase
    end

    assign a_ready = !d_valid;

    always_ff @(posedge clk) begin
        if (rst) begin
            d_valid      <= 1'b0;
            scratch      <= '0;
            denied_count <= '0;
            cycles       <= '0;
        end else begin
            cycles <= cycles + 1'b1;
            if (denied) denied_count <= denied_count + 1'b1;
            if (d_valid && d_ready) d_valid <= 1'b0;
            if (a_valid && a_ready) begin
                d_valid <= 1'b1;
                d       <= '{opcode: wr ? D_ACK : D_ACK_DATA, denied: !hit, data: wr ? '0 : rdata};
                if (wr && word == 10'h1) scratch <= apply_mask(scratch, a.data, a.mask);
            end
        end
    end
endmodule
