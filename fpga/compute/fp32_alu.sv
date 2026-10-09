// Multi-cycle FP32 ALU for the vector_op_e operations ADD, SUB, MUL, MAX, EXP2 and RECIP. Other
// operations give NaN. Normalization, rounding and products are registered, with one multiplier.
// Results equal the fp32_pkg reference functions fp_add (SUB: x + -y), fp_mul, fp_max, fp_exp2, fp_recip.
module fp32_alu
  import vector_pkg::*;
(
  input logic clk, rst, in_valid,
  output logic in_ready,
  input vector_op_e operation,
  input logic [31:0] x, y,
  output logic out_valid,
  input logic out_ready,
  output logic [31:0] result
);
  import fp32_pkg::*;
  typedef enum logic [4:0] {
    S_IDLE, S_PREPARE, S_ALIGN, S_ADD, S_PRODUCT, S_MUL_PACK,
    S_EXP_INIT, S_EXP_FOLD, S_EXP_STEP, S_EXP_PACK,
    S_RECIP_INIT, S_RECIP_CORRECT, S_RECIP_REFINE,
    S_NORM, S_SHIFT, S_ROUND, S_DONE
  } state_e;
  state_e state, after_product;
  vector_op_e op;
  logic [31:0] a, b;
  logic sign_bit;
  logic [27:0] sig_a, sig_b;
  int exponent, difference, unbiased, norm_shift;
  logic [47:0] magnitude;
  logic [24:0] significand;
  logic increment, subnormal;
  logic [31:0] multiply_a, multiply_b;
  (* use_dsp = "yes" *) logic [63:0] product;
  logic signed [63:0] fixed_x;
  logic [31:0] residual, base, polynomial, mantissa, estimate;
  logic [1:0] iteration;
  logic [31:0] operand_a, operand_b;

  assign in_ready = state == S_IDLE;
  assign out_valid = state == S_DONE;
  assign operand_a = ftz(a);
  assign operand_b = ftz(b);

  always_ff @(posedge clk) begin
    if (rst) state <= S_IDLE;
    else case (state)
      S_IDLE: if (in_valid) begin
        a <= x;
        b <= operation == V_SUB ? {~y[31], y[30:0]} : y;
        op <= operation;
        state <= S_PREPARE;
      end
      S_PREPARE: begin : prepare
        logic [31:0] first, second;
        int raw_exponent;
        first = operand_a; second = operand_b;
        state <= S_DONE;
        if (op == V_MAX) result <= fp_max(first, second);
        else if (is_nan(first) || ((op == V_ADD || op == V_SUB || op == V_MUL) && is_nan(second)))
          result <= QNAN;
        else if (op == V_ADD || op == V_SUB) begin
          if (first[30:23] == 255 || second[30:23] == 255) begin
            result <= first[30:23] == 255 ? first : second;
            if (first[30:23] == 255 && second[30:23] == 255 && first[31] != second[31]) result <= QNAN;
          end else if (first[30:0] == 0 && second[30:0] == 0) result <= {first[31] & second[31], 31'd0};
          else if (first[30:0] == 0) result <= second;
          else if (second[30:0] == 0) result <= first;
          else begin
            if (first[30:0] < second[30:0]) begin first = operand_b; second = operand_a; end
            sign_bit <= first[31];
            a <= first; b <= second;
            sig_a <= {1'b0, 1'b1, first[22:0], 3'd0};
            sig_b <= {1'b0, 1'b1, second[22:0], 3'd0};
            exponent <= int'(first[30:23]) - 153;
            difference <= int'(first[30:23]) - int'(second[30:23]);
            state <= S_ALIGN;
          end
        end else if (op == V_MUL) begin
          sign_bit <= first[31] ^ second[31];
          if (first[30:23] == 255 || second[30:23] == 255) begin
            result <= first[30:0] == 0 || second[30:0] == 0 ? QNAN : {first[31]^second[31],8'hff,23'd0};
          end else if (first[30:0] == 0 || second[30:0] == 0) result <= {first[31]^second[31],31'd0};
          else begin
            multiply_a <= {8'd0,1'b1,first[22:0]};
            multiply_b <= {8'd0,1'b1,second[22:0]};
            exponent <= int'(first[30:23]) + int'(second[30:23]) - 300;
            after_product <= S_MUL_PACK;
            state <= S_PRODUCT;
          end
        end else if (op == V_EXP2) begin
          sign_bit <= 0;
          if (first[30:0] == 0) result <= ONE;
          else if (first[30:0] >= 31'h43000000) result <= first[31] ? 32'd0 : 32'h7f800000;
          else begin
            raw_exponent = int'(first[30:23]) - 127;
            fixed_x <= raw_exponent >= -1 ?
              $signed({40'd0,1'b1,first[22:0]}) <<< (raw_exponent + 1) :
              $signed({40'd0,1'b1,first[22:0]}) >>> (-raw_exponent - 1);
            state <= S_EXP_INIT;
          end
        end else if (op == V_RECIP) begin
          sign_bit <= first[31];
          if (first[30:0] == 0) result <= {first[31],8'hff,23'd0};
          else if (first[30:23] == 255) result <= {first[31],31'd0};
          else begin
            mantissa <= {1'b0,1'b1,first[22:0],7'd0};
            estimate <= recip_seed(int'(first[22:16]));
            exponent <= 97 - int'(first[30:23]);
            iteration <= 0;
            state <= S_RECIP_INIT;
          end
        end else result <= QNAN;
      end
      S_ALIGN: begin
        if (difference >= 28) sig_b <= 28'(|sig_b);
        else if (difference > 0)
          sig_b <= (sig_b >> difference) | 28'((sig_b & ((28'd1 << difference)-1)) != 28'd0);
        state <= S_ADD;
      end
      S_ADD: begin
        magnitude <= {20'd0, a[31] == b[31] ? sig_a + sig_b : sig_a - sig_b};
        if (a[31] != b[31] && sig_a == sig_b) sign_bit <= 0;
        state <= S_NORM;
      end
      S_PRODUCT: begin
        product <= multiply_a * multiply_b;
        state <= after_product;
      end
      S_MUL_PACK: begin magnitude <= product[47:0]; state <= S_NORM; end
      S_EXP_INIT: begin : exp_init
        logic signed [63:0] signed_fixed;
        signed_fixed = a[31] ? -fixed_x : fixed_x;
        exponent <= int'(signed_fixed >>> 24) - 31;
        residual <= {7'd0,signed_fixed[17:0],7'd0};
        base <= exp2_base(int'(signed_fixed[23:18]));
        polynomial <= 119194166;
        iteration <= 0;
        state <= S_EXP_STEP;
      end
      S_EXP_STEP: begin
        multiply_a <= polynomial;
        multiply_b <= residual;
        after_product <= S_EXP_FOLD;
        state <= S_PRODUCT;
      end
      S_EXP_FOLD: begin : exp_fold
        logic [31:0] next_polynomial;
        next_polynomial = 32'((product + 64'h40000000) >> 31) +
                          (iteration == 0 ? 32'd515882496 : iteration == 1 ? 32'd1488522236 : 32'h80000000);
        polynomial <= next_polynomial;
        if (iteration == 2) begin
          multiply_a <= base;
          multiply_b <= next_polynomial;
          after_product <= S_EXP_PACK;
          state <= S_PRODUCT;
        end else begin iteration <= iteration + 1'b1; state <= S_EXP_STEP; end
      end
      S_EXP_PACK: begin
        magnitude <= 48'((product + 64'h40000000) >> 31);
        state <= S_NORM;
      end
      S_RECIP_INIT: begin
        multiply_a <= mantissa; multiply_b <= estimate;
        after_product <= S_RECIP_CORRECT;
        state <= S_PRODUCT;
      end
      S_RECIP_CORRECT: begin
        multiply_a <= estimate;
        multiply_b <= 32'h80000000 - 32'((product + 64'h20000000) >> 30);
        after_product <= S_RECIP_REFINE;
        state <= S_PRODUCT;
      end
      S_RECIP_REFINE: begin : refine
        logic [31:0] next_estimate;
        next_estimate = 32'((product + 64'h20000000) >> 30);
        estimate <= next_estimate;
        if (iteration == 1) begin magnitude <= {16'd0,next_estimate}; state <= S_NORM; end
        else begin iteration <= 1; state <= S_RECIP_INIT; end
      end
      S_NORM: begin : normalize
        int leading;
        leading = 0;
        for (int bit_index = 0; bit_index < 48; bit_index++) if (magnitude[bit_index]) leading = bit_index;
        unbiased <= exponent + leading;
        subnormal <= exponent + leading < -126;
        norm_shift <= exponent + leading < -126 ? -149-exponent : leading-23;
        if (magnitude == 0) begin result <= {sign_bit,31'd0}; state <= S_DONE; end
        else state <= S_SHIFT;
      end
      S_SHIFT: begin : shift
        logic [47:0] remainder, halfway;
        increment <= 0;
        significand <= magnitude[24:0];
        if (norm_shift > 0 && norm_shift < 48) begin
          significand <= 25'(magnitude >> norm_shift);
          remainder = magnitude & ((48'd1 << norm_shift)-1);
          halfway = 48'd1 << (norm_shift-1);
          increment <= remainder > halfway || (remainder == halfway && magnitude[norm_shift]);
        end else if (norm_shift >= 48) significand <= 0;
        else if (norm_shift < 0) significand <= 25'(magnitude << (-norm_shift));
        state <= S_ROUND;
      end
      S_ROUND: begin : round_result
        logic [24:0] rounded;
        int final_exponent;
        rounded = significand + 25'(increment);
        final_exponent = subnormal ? -126 : unbiased;
        if (rounded[24]) begin rounded = rounded >> 1; final_exponent++; end
        if (subnormal && rounded < 25'h0800000) result <= {sign_bit,31'd0};
        else if (final_exponent > 127) result <= {sign_bit,8'hff,23'd0};
        else result <= {sign_bit,8'(final_exponent+127),rounded[22:0]};
        state <= S_DONE;
      end
      S_DONE: if (out_ready) state <= S_IDLE;
      default: state <= S_IDLE;
    endcase
  end
endmodule
