// One lane of flash_vector_unit: the per-element steps of an operation, on the lane's own FP32 ALU.
// MAX and SUM pass x through (flash_vector_unit does the reduction). Disabled lanes give 0.
module vector_lane
  import vector_pkg::*;
  import fp32_pkg::*;
(
  input  logic        clk, rst,
  input  logic        start,    // begin; the inputs below hold until busy falls
  input  vector_op_e  op,
  input  logic        enable,   // lane mask bit
  input  logic [31:0] x, y,
  input  logic [31:0] factor,   // multiplier of SCALE, NORMALIZE, AXPY and SOFTMAX
  input  logic [31:0] new_max,  // row maximum of SOFTMAX
  output logic        busy,
  output logic [31:0] result
);
  typedef enum logic [1:0] {L_IDLE, L_REQ, L_WAIT} state_e;
  state_e state;
  logic [2:0] phase, iteration;
  logic [31:0] intermediate, estimate, half_mantissa;
  logic [31:0] normalized_x, rsqrt_scale;
  int rsqrt_exponent;
  vector_op_e alu_op;
  logic [31:0] alu_x, alu_y, alu_result;
  logic alu_ready, alu_complete;

  assign busy = state != L_IDLE;

  always_comb begin
    rsqrt_exponent = int'(x[30:23]) - 127;
    normalized_x = {1'b0, 8'(127 + (rsqrt_exponent & 1)), x[22:0]};
    rsqrt_scale = {1'b0, 8'(127 - (rsqrt_exponent >>> 1)), 23'd0};
  end

  // Lanes that need no ALU: disabled, ABS, reductions, empty SOFTMAX rows, RSQRT specials.
  wire rsqrt_special = x[31] || x[30:23] == 0 || x[30:23] == 255;
  wire run = enable && op != V_ABS && op != V_MAX && op != V_SUM &&
             !(op == V_SOFTMAX && new_max == NEG_INF) && !(op == V_RSQRT && rsqrt_special);

  fp32_alu u_alu (.clk, .rst, .in_valid(state == L_REQ), .in_ready(alu_ready),
    .operation(alu_op), .x(alu_x), .y(alu_y),
    .out_valid(alu_complete), .out_ready(1'b1), .result(alu_result));

  always_comb begin
    alu_op = V_ADD;
    alu_x = x;
    alu_y = y;
    case (op)
      V_ADD, V_SUB, V_MUL, V_EXP2, V_RECIP: alu_op = op;
      V_SCALE, V_NORMALIZE: begin alu_op = V_MUL; alu_y = factor; end
      V_AXPY: begin  // x * factor + y
        alu_op = phase == 0 ? V_MUL : V_ADD;
        alu_x = phase == 0 ? x : intermediate;
        alu_y = phase == 0 ? factor : y;
      end
      V_SOFTMAX: begin  // exp2((x - new_max) * factor)
        alu_op = phase == 0 ? V_SUB : phase == 1 ? V_MUL : V_EXP2;
        alu_x = phase == 0 ? x : intermediate;
        alu_y = phase == 0 ? new_max : factor;
      end
      V_RSQRT: begin
        // 4 Newton steps e = e (1.5 - 0.5 m e^2) on m in [1, 4), then e x 2^-floor(exponent / 2)
        case (phase)
          0: begin alu_op = V_MUL; alu_x = normalized_x; alu_y = 32'h3f000000; end
          1: begin alu_op = V_MUL; alu_x = estimate; alu_y = estimate; end
          2: begin alu_op = V_MUL; alu_x = half_mantissa; alu_y = intermediate; end
          3: begin alu_op = V_SUB; alu_x = 32'h3fc00000; alu_y = intermediate; end
          4: begin alu_op = V_MUL; alu_x = estimate; alu_y = intermediate; end
          default: begin alu_op = V_MUL; alu_x = estimate; alu_y = rsqrt_scale; end
        endcase
      end
      V_SILU: begin  // x * recip(1 + exp2(x * -log2(e)))
        alu_x = intermediate;
        case (phase)
          0: begin alu_op = V_MUL; alu_x = x; alu_y = 32'hbfb8aa3b; end
          1: alu_op = V_EXP2;
          2: begin alu_op = V_ADD; alu_y = ONE; end
          3: alu_op = V_RECIP;
          default: begin alu_op = V_MUL; alu_y = x; end
        endcase
      end
      default: begin alu_op = V_ADD; alu_x = QNAN; end
    endcase
  end

  always_ff @(posedge clk) begin
    if (rst) state <= L_IDLE;
    else case (state)
      L_IDLE: if (start) begin
        phase <= 0;
        iteration <= 0;
        if (run) state <= L_REQ;
        else begin
          result <= 0;
          if (enable && (op == V_MAX || op == V_SUM)) result <= x;
          if (enable && op == V_ABS) result <= x[30:23] == 0 ? 32'd0 : {1'b0, x[30:0]};
          if (enable && op == V_RSQRT) begin
            if (x[30:23] == 0) result <= {x[31], 8'hff, 23'd0};
            else if (x[31] || is_nan(x)) result <= QNAN;
          end
        end
      end
      L_REQ: if (alu_ready) state <= L_WAIT;
      L_WAIT: if (alu_complete) begin
        state <= L_REQ;
        if (op == V_RSQRT && phase < 5) begin
          if (phase == 0) begin
            half_mantissa <= alu_result;
            estimate <= 32'h5f375a86 - (normalized_x >> 1);
            phase <= 1;
          end else if (phase == 4) begin
            estimate <= alu_result;
            if (iteration == 3) phase <= 5;
            else begin iteration <= iteration + 1'b1; phase <= 1; end
          end else begin intermediate <= alu_result; phase <= phase + 1'b1; end
        end else if ((op == V_SOFTMAX && phase < 2) || (op == V_AXPY && phase == 0) ||
                     (op == V_SILU && phase < 4)) begin
          intermediate <= alu_result;
          phase <= phase + 1'b1;
        end else begin
          result <= alu_result;
          state <= L_IDLE;
        end
      end
      default: state <= L_IDLE;
    endcase
  end
endmodule
