// Scratch memory: 128 KiB of block RAM in 32-byte beats, with two ports.
// Bus port: any whole words of one beat per request; a partial word is denied.
// Vector port: one beat per cycle, any of its words written; read data one cycle later.
module scratch_memory
  import tl_pkg::*;
#(parameter int BEATS = 4096) (
  input  logic clk, rst,
  input  logic a_valid,
  output logic a_ready,
  input  wa_t  a,
  output logic d_valid,
  input  logic d_ready,
  output wd_t  d,
  input  logic                     v_en,
  input  logic                     v_we,
  input  logic [7:0]               v_wmask,
  input  logic [$clog2(BEATS)-1:0] v_addr,
  input  logic [255:0]             v_wdata,
  output logic [255:0]             v_rdata
);
  localparam int AW = $clog2(BEATS);
  logic pending;
  logic [255:0] rdata;
  logic whole;
  always_comb begin
    whole = a.mask != 0;
    for (int w = 0; w < 8; w++) if (a.mask[4*w +: 4] != 4'h0 && a.mask[4*w +: 4] != 4'hf) whole = 0;
  end
  wire valid_access = whole && (a.opcode == A_GET || a.opcode == A_PUT_FULL);
  wire [7:0] words = words_of(a.mask);
  assign a_ready = !d_valid && !pending;
  wire fire = a_valid && a_ready;

  for (genvar w = 0; w < 8; w++) begin : g_lane
    tdp_ram #(.DEPTH(BEATS)) u_mem (
      .clk,
      .a_en(fire && valid_access), .a_we(a.opcode == A_PUT_FULL && words[w]),
      .a_addr(a.address[AW+4:5]), .a_wdata(a.data[32*w +: 32]), .a_rdata(rdata[32*w +: 32]),
      .b_en(v_en), .b_we(v_we && v_wmask[w]), .b_addr(v_addr), .b_wdata(v_wdata[32*w +: 32]),
      .b_rdata(v_rdata[32*w +: 32]));
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      pending <= 0;
      d_valid <= 0;
    end else begin
      if (d_valid && d_ready) d_valid <= 0;
      if (pending) begin
        d_valid <= 1;
        d.data <= rdata;
        pending <= 0;
      end
      if (fire) begin
        d <= '{opcode: a.opcode == A_GET ? D_ACK_DATA : D_ACK, denied: !valid_access, data: '0};
        if (valid_access && a.opcode == A_GET) pending <= 1;
        else d_valid <= 1;
      end
    end
  end
endmodule
