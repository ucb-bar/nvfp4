// Kernel engine: a programmable sequencer with 16 integer registers, and the second bus master.
// Instruction: opcode[63:56], rd[55:52], rs1[51:48], rs2[47:44], immediate[31:0]. Opcodes:
//   0 HALT, 1 MOVI, 2 ADDI, 3 ADD, 4 LOAD, 5 STORE, 6 JMP, 7 JNZ, 8 COPY and 9 FILL (immediate
//   words), 10 WAIT (error mask [31:16], busy mask [15:0]), 11 SHL, 12 SHR and 13 AND (with the
//   immediate, shifts by immediate[4:0]), 14 SHL_REG (shift by rs2[4:0]),
//   15 SLTU (rd = rs1 < rs2, unsigned).
// Faults: 1 unknown opcode, 2 PC outside the program, 3 denied access, 4 WAIT error bit set.
// MMIO: 0x00 ID, 0x04 capacity, 0x08 status {halted, fault, busy}, 0x0C start, 0x10 entry,
//   0x14 length, 0x18 cycles, 0x1C PC, 0x20 fault, 0x100 integer registers [16],
//   0x1000 program (8192 little-endian 64-bit instructions).
// The master port moves 32-byte beats. A COPY step moves the words up to the next beat boundary of
// the source or destination, so equally aligned buffers take one read and one write per 8 words.
module kernel_engine
  import tl_pkg::*;
(
  input logic clk, rst,
  input logic a_valid,
  output logic a_ready,
  input a_t a,
  output logic d_valid,
  input logic d_ready,
  output d_t d,
  output logic master_a_valid,
  input logic master_a_ready,
  output wa_t master_a,
  input logic master_d_valid,
  output logic master_d_ready,
  input wd_t master_d,
  output logic busy
);
  typedef enum logic [2:0] {IDLE, FETCH, EXECUTE, REQUEST, RESPONSE} state_e;
  state_e state;
  logic [31:0] regs [16];
  logic [31:0] entry, length, pc, cycles, fault;
  logic done;
  logic [31:0] code_lo, code_hi;
  wire [63:0] instruction = {code_hi, code_lo};
  wire [7:0] op = instruction[63:56];
  wire [3:0] rd = instruction[55:52], rs1 = instruction[51:48], rs2 = instruction[47:44];
  wire [31:0] imm = instruction[31:0];
  logic [7:0] active_op;
  logic [3:0] active_rd;
  logic [31:0] src, dst, count, fill_value, wait_mask;
  logic [3:0] step;     // words of the current COPY/FILL step
  logic [2:0] lane;     // word lane of a LOAD or WAIT
  logic copying_write;
  logic [31:0] rdata;
  logic mapped, writable;
  wire [16:0] off = a.address[16:0];
  wire program_sel = off >= 17'h1000 && off < 17'h11000;
  wire reg_sel = off >= 17'h100 && off < 17'h140;
  wire wr = a.opcode != A_GET;
  assign busy = state != IDLE;
  assign a_ready = !d_valid;
  wire fire = a_valid && a_ready;
  wire allowed_write = a.opcode == A_PUT_FULL && a.mask == 4'hf && writable && !busy;
  // Words to the next beat boundary of both addresses, at most n
  function automatic logic [3:0] span(logic [31:0] n, logic [31:0] p, logic [31:0] q);
    logic [3:0] k;
    k = 4'd8 - 4'(p[4:2]);
    if (4'd8 - 4'(q[4:2]) < k) k = 4'd8 - 4'(q[4:2]);
    if (n < 32'(k)) k = 4'(n);
    return k;
  endfunction
  // Beat with word i moved to word i + shift (mod 8)
  function automatic logic [255:0] rotate(logic [255:0] data, logic [2:0] shift);
    for (int i = 0; i < 8; i++) rotate[32 * 3'(i + 32'(shift)) +: 32] = data[32 * i +: 32];
  endfunction
  sdp_ram #(.DEPTH(8192)) u_code_lo (
    .clk, .we(fire && allowed_write && program_sel && !off[2]),
    .waddr(off[15:3] - 13'h200), .wdata(a.data),
    .re(state == FETCH), .raddr(pc[12:0]), .rdata(code_lo));
  sdp_ram #(.DEPTH(8192)) u_code_hi (
    .clk, .we(fire && allowed_write && program_sel && off[2]),
    .waddr(off[15:3] - 13'h200), .wdata(a.data),
    .re(state == FETCH), .raddr(pc[12:0]), .rdata(code_hi));
  always_comb begin
    rdata = 0;
    mapped = off[1:0] == 0;
    writable = 0;
    case (off)
      17'h00: rdata = 32'h4b45_524e; // KERN
      17'h04: rdata = 8192;
      17'h08: rdata = {29'd0, done, fault != 0, busy};
      17'h0c: writable = 1;
      17'h10: begin rdata = entry; writable = 1; end
      17'h14: begin rdata = length; writable = 1; end
      17'h18: rdata = cycles;
      17'h1c: rdata = pc;
      17'h20: rdata = fault;
      default: begin
        if (reg_sel) begin rdata = regs[off[5:2]]; writable = 1; end
        else if (program_sel && wr) writable = 1;
        else mapped = 0;
      end
    endcase
  end
  assign master_a_valid = state == REQUEST;
  assign master_d_ready = state == RESPONSE;
  always_ff @(posedge clk) begin
    if (rst) begin
      state <= IDLE;
      d_valid <= 0;
      entry <= 0;
      length <= 0;
      pc <= 0;
      cycles <= 0;
      fault <= 0;
      done <= 0;
      for (int i = 0; i < 16; i++) regs[i] <= 0;
    end else begin
      if (d_valid && d_ready) d_valid <= 0;
      if (busy) cycles <= cycles + 1;
      if (fire) begin
        d_valid <= 1;
        d <= '{opcode: wr ? D_ACK : D_ACK_DATA,
               denied: !mapped || (wr && !allowed_write) ||
                       (!wr && a.opcode != A_GET), data: wr ? 32'd0 : rdata};
        if (mapped && allowed_write) begin
          case (off)
            17'h0c: if (a.data[0]) begin
              state <= FETCH;
              pc <= entry;
              cycles <= 0;
              fault <= 0;
              done <= 0;
            end
            17'h10: entry <= a.data;
            17'h14: length <= a.data;
            default: if (reg_sel) regs[off[5:2]] <= a.data;
          endcase
        end
      end
      case (state)
        FETCH: begin
          if (pc >= length || length > 8192) begin
            fault <= 2;
            state <= IDLE;
          end else state <= EXECUTE;
        end
        EXECUTE: begin
          pc <= pc + 1;
          state <= FETCH;
          active_op <= op;
          active_rd <= rd;
          copying_write <= 0;
          case (op)
            0: begin state <= IDLE; done <= 1; end
            1: regs[rd] <= imm;
            2: regs[rd] <= regs[rs1] + imm;
            3: regs[rd] <= regs[rs1] + regs[rs2];
            4, 5: begin : load_store
              logic [31:0] address;
              address = regs[rs1] + imm;
              lane <= address[4:2];
              master_a <= '{opcode: op == 4 ? A_GET : A_PUT_FULL, address: address,
                            mask: word_span(address[4:2], 4'd1), data: {8{regs[rs2]}}};
              state <= REQUEST;
            end
            6: pc <= imm;
            7: if (regs[rs1] != 0) pc <= imm;
            11: regs[rd] <= regs[rs1] << imm[4:0];
            12: regs[rd] <= regs[rs1] >> imm[4:0];
            13: regs[rd] <= regs[rs1] & imm;
            14: regs[rd] <= regs[rs1] << regs[rs2][4:0];
            15: regs[rd] <= 32'(regs[rs1] < regs[rs2]);
            8, 9: if (imm != 0) begin : transfer
              logic [3:0] n;
              n = span(imm, regs[rs1], op == 8 ? regs[rs2] : regs[rs1]);
              src <= regs[rs1];
              dst <= op == 8 ? regs[rs2] : regs[rs1];
              count <= imm;
              step <= n;
              fill_value <= regs[rs2];
              master_a <= '{opcode: op == 8 ? A_GET : A_PUT_FULL, address: regs[rs1],
                            mask: word_span(regs[rs1][4:2], n), data: {8{regs[rs2]}}};
              state <= REQUEST;
            end
            10: begin
              wait_mask <= imm;
              lane <= regs[rs1][4:2];
              master_a <= '{opcode: A_GET, address: regs[rs1], mask: word_span(regs[rs1][4:2], 4'd1),
                            data: '0};
              state <= REQUEST;
            end
            default: begin fault <= 1; state <= IDLE; end
          endcase
        end
        REQUEST: if (master_a_ready) state <= RESPONSE;
        RESPONSE: if (master_d_valid) begin
          state <= FETCH;
          if (master_d.denied) begin fault <= 3; state <= IDLE; end
          else case (active_op)
            4: regs[active_rd] <= master_d.data[32 * lane +: 32];
            8: if (!copying_write) begin
              master_a <= '{opcode: A_PUT_FULL, address: dst, mask: word_span(dst[4:2], step),
                            data: rotate(master_d.data, dst[4:2] - src[4:2])};
              copying_write <= 1;
              state <= REQUEST;
            end else if (count > 32'(step)) begin : next_copy
              logic [31:0] next_src, next_dst;
              logic [3:0] n;
              next_src = src + 4 * 32'(step);
              next_dst = dst + 4 * 32'(step);
              n = span(count - 32'(step), next_src, next_dst);
              src <= next_src;
              dst <= next_dst;
              count <= count - 32'(step);
              step <= n;
              master_a <= '{opcode: A_GET, address: next_src, mask: word_span(next_src[4:2], n),
                            data: '0};
              copying_write <= 0;
              state <= REQUEST;
            end
            9: if (count > 32'(step)) begin : next_fill
              logic [31:0] next_dst;
              logic [3:0] n;
              next_dst = dst + 4 * 32'(step);
              n = span(count - 32'(step), next_dst, next_dst);
              dst <= next_dst;
              count <= count - 32'(step);
              step <= n;
              master_a <= '{opcode: A_PUT_FULL, address: next_dst, mask: word_span(next_dst[4:2], n),
                            data: {8{fill_value}}};
              state <= REQUEST;
            end
            10: begin : wait_poll
              logic [31:0] status;
              status = master_d.data[32 * lane +: 32];
              if ((status & {16'd0, wait_mask[31:16]}) != 0) begin
                fault <= 4;
                state <= IDLE;
              end else if ((status & {16'd0, wait_mask[15:0]}) != 0) state <= REQUEST;
            end
            default: ;
          endcase
        end
        default: ;
      endcase
    end
  end
endmodule
