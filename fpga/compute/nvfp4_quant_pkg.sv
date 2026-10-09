// Finite NVFP4 conversion: UE4M3 block scale and E2M1 elements, nearest-even ties.
package nvfp4_quant_pkg;
  // UE4M3 code of |value|. Zero and subnormal values give 0. Values from 448 up saturate to 448.
  function automatic logic [7:0] encode_scale(input logic [31:0] value);
    int exponent, shift;
    logic [31:0] significand, rounded, remainder, halfway;
    if (value[30:23] == 0) return 0;
    if (value[30:0] >= 31'h43e00000) return 126; // saturate at 448
    exponent = int'(value[30:23]) - 127;
    significand = {8'd0, 1'b1, value[22:0]};
    shift = exponent < -6 ? 14 - exponent : 20;
    if (shift > 31) return 0;
    rounded = significand >> shift;
    remainder = significand & ((32'd1 << shift) - 1);
    halfway = 32'd1 << (shift - 1);
    rounded = rounded + 32'(remainder > halfway || (remainder == halfway && rounded[0]));
    if (exponent < -6) return 8'(rounded);
    if (rounded == 16) begin rounded = 8; exponent++; end
    return 8'(((exponent + 7) << 3) | int'(rounded[2:0]));
  endfunction

  // Midpoint `boundary` (0-6) between adjacent E2M1 magnitudes, times scale. The product is exact
  // in FP32 for every finite UE4M3 scale.
  function automatic logic [31:0] threshold(input logic [7:0] scale, input int boundary);
    int significand, exponent, numerator, product, leading;
    case (boundary)
      0: numerator = 1;
      1: numerator = 3;
      2: numerator = 5;
      3: numerator = 7;
      4: numerator = 10;
      5: numerator = 14;
      default: numerator = 20;
    endcase
    significand = scale[6:3] == 0 ? int'(scale[2:0]) : 8 + int'(scale[2:0]);
    exponent = scale[6:3] == 0 ? -9 : int'(scale[6:3]) - 10;
    product = significand * numerator;
    leading = 0;
    for (int i = 0; i < 9; i++) if ((product & (1 << i)) != 0) leading = i;
    if (product == 0) return 0;
    return (32'(exponent - 2 + leading + 127) << 23) |
           (32'(product << (23 - leading)) & 32'h7fffff);
  endfunction

  // E2M1 code of value / scale, given the 7 thresholds of scale. A zero scale gives code 0.
  function automatic logic [3:0] encode_element(input logic [31:0] value, input logic [31:0] limit [7],
                                                input logic [7:0] scale);
    logic [2:0] code;
    logic [31:0] magnitude;
    magnitude = value[30:23] == 0 ? 32'd0 : {1'b0, value[30:0]};
    code = 0;
    if (scale != 0)
      for (int t = 0; t < 7; t++)
        if (magnitude > limit[t] || (magnitude == limit[t] && (t & 1) != 0)) code = 3'(t + 1);
    return {value[31], code};
  endfunction
endpackage
