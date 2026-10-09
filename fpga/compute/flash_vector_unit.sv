// Eight-lane SIMD vector unit, one command in flight. Each lane has its own FP32 ALU for the
// per-element steps. One more ALU runs the steps that combine lanes, in lane order:
//   MAX, SUM      reduction over the masked lanes, starting from -inf or 0
//   SOFTMAX       m_new = max(m_old, masked x); p = exp2((x - m_new) * scalar);
//                 alpha = exp2((m_old - m_new) * scalar), or 0 if l_old is 0;
//                 l_new = alpha * l_old + (sum of p in lane order), for O_new = alpha * O_old + P V
//   NORMALIZE     x * recip(scalar), or x * 0 if scalar is 0
// The caller owns the per-row (m, l, O); the lane mask excludes entries from both reductions.
module flash_vector_unit
  import vector_pkg::*;
#(parameter int LANES = 8) (
  input logic clk, rst,
  input logic in_valid,
  output logic in_ready,
  input vector_op_e operation,
  input logic [LANES-1:0] lane_mask,
  input logic [31:0] x [LANES], y [LANES],
  input logic [31:0] scalar, old_max, old_sum,
  output logic out_valid,
  input logic out_ready,
  output logic [31:0] result [LANES],
  output logic [31:0] reduction, new_max, new_sum, alpha
);
  import fp32_pkg::*;
  localparam int LW = LANES > 1 ? $clog2(LANES) : 1;
  // CHAIN: acc = ALU(acc, value[lane]) over chain lanes. SEQ: a fixed sequence of ALU steps.
  typedef enum logic [2:0] {S_IDLE, S_CHAIN, S_SEQ, S_START, S_LANES, S_DONE} state_e;
  typedef enum logic [1:0] {C_MAX, C_REDUCE, C_ACC} chain_e;
  typedef enum logic [1:0] {Q_RECIP, Q_ALPHA, Q_FINAL} seq_e;
  state_e state;
  chain_e chain;
  seq_e seq;
  vector_op_e op;
  logic [LW-1:0] lane;
  logic [1:0] phase;
  logic waiting;  // the combining ALU holds a step
  logic [LANES-1:0] mask, ran, lane_busy;
  logic [31:0] vx [LANES], vy [LANES];
  logic [31:0] factor, previous_max, previous_sum, running, intermediate;
  logic alu_valid, alu_ready, alu_complete;
  vector_op_e alu_op;
  logic [31:0] alu_x, alu_y, alu_result;

  assign in_ready = state == S_IDLE;
  assign out_valid = state == S_DONE;

  for (genvar i = 0; i < LANES; i++) begin : g_lane
    vector_lane u_lane (
      .clk, .rst, .start(state == S_START), .op, .enable(mask[i]), .x(vx[i]), .y(vy[i]),
      .factor, .new_max, .busy(lane_busy[i]), .result(result[i]));
  end

  fp32_alu u_alu (.clk, .rst, .in_valid(alu_valid), .in_ready(alu_ready),
    .operation(alu_op), .x(alu_x), .y(alu_y),
    .out_valid(alu_complete), .out_ready(1'b1), .result(alu_result));

  // Lanes that take part in the current chain
  logic [LANES-1:0] chain_lanes;
  assign chain_lanes = chain == C_ACC ? ran : mask;

  always_comb begin
    alu_valid = 0;
    alu_op = V_ADD;
    alu_x = 0;
    alu_y = 0;
    if (state == S_CHAIN) begin
      alu_valid = !waiting && chain_lanes[lane];
      case (chain)
        C_MAX: begin alu_op = V_MAX; alu_x = new_max; alu_y = vx[lane]; end
        C_REDUCE: begin alu_op = op == V_MAX ? V_MAX : V_ADD; alu_x = reduction; alu_y = vx[lane]; end
        default: begin alu_op = V_ADD; alu_x = running; alu_y = result[lane]; end
      endcase
    end else if (state == S_SEQ || (state == S_LANES && seq == Q_ALPHA && phase != 3)) begin
      alu_valid = !waiting;
      case (seq)
        Q_RECIP: begin alu_op = V_RECIP; alu_x = factor; end
        Q_ALPHA: begin
          alu_op = phase == 0 ? V_SUB : phase == 1 ? V_MUL : V_EXP2;
          alu_x = phase == 0 ? previous_max : intermediate;
          alu_y = phase == 0 ? new_max : factor;
        end
        default: begin
          alu_op = phase == 0 ? V_MUL : V_ADD;
          alu_x = phase == 0 ? alpha : intermediate;
          alu_y = phase == 0 ? previous_sum : running;
        end
      endcase
    end
  end

  always_ff @(posedge clk) begin
    if (rst) state <= S_IDLE;
    else begin
      if (alu_valid && alu_ready) waiting <= 1;
      case (state)
        S_IDLE: if (in_valid) begin
          op <= operation; mask <= lane_mask; vx <= x; vy <= y;
          factor <= operation == V_NORMALIZE && scalar[30:0] == 0 ? 32'd0 : scalar;
          previous_max <= old_max; previous_sum <= old_sum;
          new_max <= old_sum[30:0] == 0 ? NEG_INF : old_max;
          new_sum <= 0; alpha <= ONE; running <= 0;
          reduction <= operation == V_MAX ? NEG_INF : 32'd0;
          lane <= 0; phase <= 0; waiting <= 0;
          seq <= Q_RECIP;
          if (operation == V_SOFTMAX) begin chain <= C_MAX; state <= S_CHAIN; end
          else if (operation == V_MAX || operation == V_SUM) begin chain <= C_REDUCE; state <= S_CHAIN; end
          else if (operation == V_NORMALIZE && scalar[30:0] != 0) state <= S_SEQ;
          else state <= S_START;
        end
        S_CHAIN: if (!chain_lanes[lane] || (waiting && alu_complete)) begin
          if (waiting) begin
            waiting <= 0;
            case (chain)
              C_MAX: new_max <= alu_result;
              C_REDUCE: reduction <= alu_result;
              default: running <= alu_result;
            endcase
          end
          if (lane != LW'(LANES - 1)) lane <= lane + 1'b1;
          else begin
            lane <= 0;
            if (chain == C_ACC) begin seq <= Q_FINAL; phase <= 0; state <= S_SEQ; end
            else state <= S_START;
          end
        end
        S_SEQ: if (waiting && alu_complete) begin
          waiting <= 0;
          if (seq == Q_RECIP) begin factor <= alu_result; state <= S_START; end
          else if (phase == 0) begin intermediate <= alu_result; phase <= 1; end
          else begin new_sum <= alu_result; state <= S_DONE; end
        end
        // Lanes start. SOFTMAX computes alpha on the combining ALU while the lanes run.
        S_START: begin
          ran <= mask & {LANES{op != V_SOFTMAX || new_max != NEG_INF}};
          phase <= 0;
          if (op == V_SOFTMAX) begin
            seq <= Q_ALPHA;
            if (previous_sum[30:0] == 0) begin alpha <= 0; phase <= 3; end
          end
          state <= S_LANES;
        end
        S_LANES: begin
          if (seq == Q_ALPHA && waiting && alu_complete) begin
            waiting <= 0;
            if (phase == 2) alpha <= alu_result;
            else intermediate <= alu_result;
            phase <= phase + 1'b1;
          end
          if (lane_busy == 0 && (op != V_SOFTMAX || phase == 3)) begin
            if (op == V_SOFTMAX) begin chain <= C_ACC; lane <= 0; state <= S_CHAIN; end
            else state <= S_DONE;
          end
        end
        S_DONE: if (out_ready) state <= S_IDLE;
        default: state <= S_IDLE;
      endcase
    end
  end
endmodule
