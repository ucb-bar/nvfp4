// Streaming vector device. A command runs one operation of flash_vector_unit on GROUPS groups of
// 8 FP32 elements: group g reads X + 32 g (and Y + 32 g) from scratch memory and writes the result
// lanes to D + 32 g, through the vector port of scratch_memory. MASK selects lanes in every group;
// unselected result lanes are written as 0, and MAX and SUM pass x through. For MAX and SUM,
// REDUCTION combines the group reductions left to right. SOFTMAX takes one group.
//
// QUANT converts GROUPS rows of MASK 16-element blocks (row r, block b at X + 64 (r MASK + b)) to
// NVFP4 with inverse tensor scale SCALAR. It writes the E2M1 codes of block b to D + 128 r + 8 b,
// low nibble first, and the UE4M3 scale bytes of row r to Y + 16 r as whole words, zero past the
// last block. These are the tensor core's operand layouts. A non-finite element or scale stops
// the command and sets rejected.
//
// 0x00 ID "FVEC", 0x04 LANES, 0x08 STATUS {rejected, busy},
// 0x10 REDUCTION, 0x14 MAX, 0x18 SUM, 0x1C ALPHA (results of the last group), read-only.
// Command, one 32-byte beat:
// 0x20 X, 0x24 Y, 0x28 D (32-byte aligned scratch addresses), 0x2C {MASK[23:16], GROUPS[15:0]},
// 0x30 SCALAR, 0x34 OLD_MAX, 0x38 OLD_SUM, 0x3C CTRL {op[7:4], start[0]}.
// Writes stall while a command runs; reads do not. Full-word writes only.
module flash_vector
  import tl_pkg::*;
  import vector_pkg::*;
  import nvfp4_quant_pkg::*;
#(parameter int BEATS = 4096) (
  input logic clk, rst,
  input logic a_valid,
  output logic a_ready,
  input a_t a,
  output logic d_valid,
  input logic d_ready,
  output d_t d,
  output logic                     v_en,
  output logic                     v_we,
  output logic [7:0]               v_wmask,  // words of the beat written
  output logic [$clog2(BEATS)-1:0] v_addr,
  output logic [255:0]             v_wdata,
  input  logic [255:0]             v_rdata,
  output logic busy
);
  localparam int LANES = 8;
  localparam int AW = $clog2(BEATS);
  localparam logic [31:0] SCRATCH = 32'h2000_0000;
  localparam logic [31:0] SIXTH = 32'h3e2aaaab;  // RNE(1/6)
  logic [31:0] cmd [8];  // the command beat
  logic [31:0] reduction, new_max, new_sum, alpha;
  logic rejected;
  logic [31:0] rdata;
  logic mapped;
  wire [15:0] off = a.address[15:0];
  wire wr = a.opcode != A_GET;
  wire full = a.mask == 4'hf;
  assign a_ready = !d_valid && !(busy && wr);
  wire fire = a_valid && a_ready;

  always_comb begin
    rdata = 0;
    mapped = off[1:0] == 0 && off < 16'h40 && off != 16'h0c;
    case (off[5:2])
      4'h0: rdata = 32'h4656_4543; // FVEC
      4'h1: rdata = LANES;
      4'h2: rdata = {30'd0, rejected, busy};
      4'h4: rdata = reduction;
      4'h5: rdata = new_max;
      4'h6: rdata = new_sum;
      4'h7: rdata = alpha;
      default: rdata = off[5] ? cmd[off[4:2]] : 0;
    endcase
  end

  wire [31:0] x_base = cmd[0], y_base = cmd[1], d_base = cmd[2];
  wire [15:0] groups = cmd[3][15:0];
  wire [7:0]  mask   = cmd[3][23:16];
  vector_op_e op;
  wire quant = op == V_QUANT;
  wire uses_y = op == V_ADD || op == V_SUB || op == V_MUL || op == V_AXPY;
  wire reduces = op == V_MAX || op == V_SUM;
  function automatic logic in_scratch(logic [31:0] address);
    return address[31:AW+5] == SCRATCH[31:AW+5] && address[4:0] == 0;
  endfunction
  // Command checks: operand addresses, group and block counts, the QUANT inverse scale.
  function automatic logic launch_ok(logic [3:0] o, logic [31:0] x, logic [31:0] y, logic [31:0] dst,
                                     logic [31:0] count, logic [31:0] scalar);
    if (o == 4'(V_QUANT))
      return count[15:0] != 0 && count[23:16] != 0 && count[23:16] <= 16 &&
             in_scratch(x) && in_scratch(y) && in_scratch(dst) &&
             !scalar[31] && scalar[30:23] != 0 && scalar[30:23] != 255;
    return o <= 4'(V_ABS) && count[15:0] != 0 && in_scratch(x) && in_scratch(dst) &&
           (!(o inside {4'(V_ADD), 4'(V_SUB), 4'(V_MUL), 4'(V_AXPY)}) || in_scratch(y)) &&
           (o != 4'(V_SOFTMAX) || count[15:0] == 1);
  endfunction
  wire launch = launch_ok(a.data[7:4], x_base, y_base, d_base, cmd[3], cmd[4]);

  // ---------------------------------------------------------------- unit
  logic unit_in_valid, unit_in_ready, unit_out_valid;
  vector_op_e unit_op;
  logic [7:0] unit_mask;
  logic [31:0] unit_x [LANES], unit_y [LANES];
  logic [31:0] unit_scalar;
  logic [31:0] unit_result [LANES];
  logic [31:0] unit_reduction, unit_max, unit_sum, unit_alpha;

  flash_vector_unit #(.LANES(LANES)) u_vector (
    .clk, .rst, .in_valid(unit_in_valid), .in_ready(unit_in_ready), .operation(unit_op),
    .lane_mask(unit_mask), .x(unit_x), .y(unit_y),
    .scalar(unit_scalar), .old_max(cmd[5]), .old_sum(cmd[6]),
    .out_valid(unit_out_valid), .out_ready(1'b1), .result(unit_result),
    .reduction(unit_reduction), .new_max(unit_max), .new_sum(unit_sum), .alpha(unit_alpha));

  // ---------------------------------------------------------------- stream
  // fetch: read x, then y, of group gf into the staging registers; issue: start the unit on them;
  // retire: write the result of group gr. A result write takes the port before a read. For MAX
  // and SUM, each group after the first is followed by a combine command: the same operation on
  // lanes {total, part}.
  logic [15:0] gf, gr;
  logic [AW-1:0] xb, yb, db;  // beats of the next reads and the next write
  logic staged, want_y, read_x, read_y, combine, unit_combine, inflight;
  logic [31:0] sx [LANES], sy [LANES];
  logic [31:0] total, part;

  wire stream  = busy && !quant;
  wire write   = stream && unit_out_valid && !unit_combine;
  wire issue_x = stream && !staged && !read_x && !want_y && !read_y && gf != groups && !write;
  wire issue_y = stream && want_y && !write;
  wire s_issue = stream && !inflight && (combine || staged);

  // ---------------------------------------------------------------- quantize
  // Block: read its two beats, then SCALE each half by the inverse scale while a scan finds the
  // largest input magnitude (zero for subnormals). The |product| of that input is
  // RNE(max x inverse); one more SCALE by RNE(1/6) gives the block scale before UE4M3 rounding.
  typedef enum logic [3:0] {Q_READ_LO, Q_READ_HI, Q_LO, Q_HI, Q_SCALE, Q_CODES, Q_WRITE, Q_ROW} qstate_e;
  qstate_e qs;
  logic q_read_hi, q_wait;
  logic [1:0] qc;                        // Q_CODES step
  logic [AW-1:0] qxb;                    // beat of the next block
  logic [15:0] qrow;
  logic [4:0] qblk;
  logic [31:0] qx [16], qp [16];
  logic [4:0] qi;                        // scan index
  logic [30:0] qmax;
  logic [3:0] qarg;
  logic qfound, qbad;
  logic [7:0] qscale;
  logic [63:0] qcodes;
  logic [127:0] qscales;                 // scale bytes of the row
  logic [31:0] qlimit [7];               // E2M1 decision thresholds of qscale

  wire q_scanned = qi == 16;
  wire [31:0] q_first = qfound ? {1'b0, qp[qarg][30:0]} : 32'd0;  // RNE(max x inverse)
  wire q_reject = q_scanned && (qbad || q_first[30:23] == 255);
  wire q_issue = quant && busy && !q_wait &&
                 (qs == Q_LO || qs == Q_HI || (qs == Q_SCALE && q_scanned && !q_reject));

  // ---------------------------------------------------------------- unit and port inputs
  wire [AW-1:0] q_code_beat  = AW'(d_base[AW+4:5]) + AW'({qrow, 2'b00}) + AW'(qblk[4:2]);
  wire [AW-1:0] q_scale_beat = AW'(y_base[AW+4:5]) + AW'(qrow[15:1]);
  wire [3:0]    q_scale_words = 4'((32'(mask) + 3) >> 2);
  assign unit_in_valid = s_issue || q_issue;
  always_comb begin
    unit_op = quant ? V_SCALE : op;
    unit_mask = quant ? (qs == Q_SCALE ? 8'h01 : 8'hff) : combine ? 8'h03 : mask;
    unit_scalar = quant && qs == Q_SCALE ? SIXTH : cmd[4];
    for (int i = 0; i < LANES; i++) begin
      unit_y[i] = sy[i];
      if (quant) unit_x[i] = qs == Q_SCALE ? (i == 0 ? q_first : 0) : qs == Q_HI ? qx[8 + i] : qx[i];
      else unit_x[i] = combine ? (i == 0 ? total : i == 1 ? part : 0) : sx[i];
    end
    v_en = write || issue_x || issue_y;
    v_we = write;
    v_wmask = 8'hff;
    v_addr = write ? db : issue_y ? yb : xb;
    for (int i = 0; i < LANES; i++) v_wdata[32*i +: 32] = unit_result[i];
    if (quant) begin
      v_en = busy && (qs == Q_READ_LO || qs == Q_READ_HI || qs == Q_WRITE || qs == Q_ROW);
      v_we = qs == Q_WRITE || qs == Q_ROW;
      v_addr = qs == Q_READ_LO ? qxb : qs == Q_READ_HI ? qxb + 1'b1 :
               qs == Q_WRITE ? q_code_beat : q_scale_beat;
      v_wmask = qs == Q_WRITE ? 8'b11 << {qblk[1:0], 1'b0} :
                8'(((32'd1 << q_scale_words) - 1) << {qrow[0], 2'b00});
      v_wdata = qs == Q_WRITE ? {4{qcodes}} : {2{qscales}};
    end
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      d_valid <= 0;
      busy <= 0;
      rejected <= 0;
      read_x <= 0;
      read_y <= 0;
      want_y <= 0;
      q_read_hi <= 0;
      for (int i = 0; i < 8; i++) cmd[i] <= 0;
    end else begin
      if (d_valid && d_ready) d_valid <= 0;
      if (fire) begin
        d_valid <= 1;
        d <= '{opcode: wr ? D_ACK : D_ACK_DATA, denied: !mapped || (wr && (!full || off < 16'h20)),
               data: wr ? 32'd0 : rdata};
        if (wr && mapped && full && off[5]) begin
          cmd[off[4:2]] <= a.data;
          if (off[4:2] == 3'd7 && a.data[0]) begin
            rejected <= !launch;
            if (launch) begin
              op <= vector_op_e'(a.data[7:4]);
              busy <= 1;
              gf <= 0;
              gr <= 0;
              xb <= x_base[AW+4:5];
              yb <= y_base[AW+4:5];
              db <= d_base[AW+4:5];
              staged <= 0;
              combine <= 0;
              inflight <= 0;
              qs <= Q_READ_LO;
              qxb <= x_base[AW+4:5];
              qrow <= 0;
              qblk <= 0;
              qscales <= 0;
              q_wait <= 0;
            end
          end
        end
      end

      // ---- stream
      // fetch: read data is at v_rdata the cycle after the read
      read_x <= issue_x;
      read_y <= issue_y;
      if (issue_y) want_y <= 0;
      if (read_x) begin
        for (int i = 0; i < LANES; i++) sx[i] <= v_rdata[32*i +: 32];
        want_y <= uses_y;
      end
      if (read_y) for (int i = 0; i < LANES; i++) sy[i] <= v_rdata[32*i +: 32];
      if ((read_x && !uses_y) || read_y) begin
        staged <= 1;
        gf <= gf + 1'b1;
        xb <= xb + 1'b1;
        yb <= yb + 1'b1;
      end
      // issue
      if (s_issue && unit_in_ready) begin
        inflight <= 1;
        unit_combine <= combine;
        if (combine) combine <= 0;
        else staged <= 0;
      end
      // retire
      if (stream && unit_out_valid) begin
        inflight <= 0;
        if (unit_combine) total <= unit_reduction;
        else begin
          gr <= gr + 1'b1;
          db <= db + 1'b1;
          new_max <= unit_max;
          new_sum <= unit_sum;
          alpha <= unit_alpha;
          if (gr == 0) total <= unit_reduction;
          else begin part <= unit_reduction; combine <= reduces; end
        end
      end
      if (stream && gr == groups && !combine && !inflight && !unit_out_valid) begin
        busy <= 0;
        reduction <= reduces ? total : unit_reduction;
      end

      // ---- quantize
      // the high beat is at v_rdata the cycle after Q_READ_HI
      q_read_hi <= quant && busy && qs == Q_READ_HI;
      if (q_read_hi) for (int i = 0; i < 8; i++) qx[8 + i] <= v_rdata[32*i +: 32];
      if (q_issue && unit_in_ready) q_wait <= 1;
      // scan one input per cycle while the halves are scaled
      if (quant && busy && (qs == Q_LO || qs == Q_HI || qs == Q_SCALE) && !q_read_hi && !q_scanned) begin
        if (qx[qi[3:0]][30:23] == 255) qbad <= 1;
        if (qx[qi[3:0]][30:23] != 0 && qx[qi[3:0]][30:0] > qmax) begin
          qmax <= qx[qi[3:0]][30:0];
          qarg <= qi[3:0];
          qfound <= 1;
        end
        qi <= qi + 1'b1;
      end
      if (quant && busy) case (qs)
        Q_READ_LO: qs <= Q_READ_HI;
        Q_READ_HI: begin
          for (int i = 0; i < 8; i++) qx[i] <= v_rdata[32*i +: 32];
          qxb <= qxb + AW'(2);
          qi <= 0;
          qmax <= 0;
          qfound <= 0;
          qbad <= 0;
          qs <= Q_LO;
        end
        Q_LO, Q_HI: if (unit_out_valid) begin
          q_wait <= 0;
          for (int i = 0; i < 8; i++) qp[qs == Q_HI ? 8 + i : i] <= unit_result[i];
          qs <= qs == Q_LO ? Q_HI : Q_SCALE;
        end
        Q_SCALE: begin
          if (q_reject && !q_wait) begin
            rejected <= 1;
            busy <= 0;
          end
          if (unit_out_valid) begin
            q_wait <= 0;
            if (unit_result[0][30:23] == 255) begin
              rejected <= 1;
              busy <= 0;
            end else begin
              qscale <= encode_scale(unit_result[0]);
              qc <= 0;
              qs <= Q_CODES;
            end
          end
        end
        Q_CODES: begin  // thresholds, then 8 codes per cycle
          if (qc == 0) for (int t = 0; t < 7; t++) qlimit[t] <= threshold(qscale, t);
          else for (int i = 0; i < 8; i++)
            qcodes[32 * int'(qc[1]) + 4 * i +: 4] <= encode_element(qp[8 * int'(qc[1]) + i], qlimit, qscale);
          qc <= qc + 1'b1;
          if (qc == 2) qs <= Q_WRITE;
        end
        Q_WRITE: begin
          qscales[8 * qblk[3:0] +: 8] <= qscale;
          if (qblk == 5'(mask - 1)) qs <= Q_ROW;
          else begin qblk <= qblk + 1'b1; qs <= Q_READ_LO; end
        end
        Q_ROW: begin
          qscales <= 0;
          qblk <= 0;
          qrow <= qrow + 1'b1;
          if (qrow == groups - 1) busy <= 0;
          else qs <= Q_READ_LO;
        end
        default: ;
      endcase
    end
  end
endmodule
