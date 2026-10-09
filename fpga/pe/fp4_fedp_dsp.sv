// 2 x N fp4_fedp dot products on shared operands, d[r][s] = C[r][s] + a_r . b_s over 64 elements,
// with the E2M1 products in DSP48E1 slices. Same arithmetic, outputs and latency as fp4_fedp.
//
// Elements are integers in units of 1/2 (|x| <= 12, products |x y| <= 144). For element k and the
// column pair (2p, 2p + 1), one slice multiplies
//   (b1 2^20 + b0) x (a1 2^10 + a0) = a1 b1 2^30 + a0 b1 2^20 + a1 b0 2^10 + a0 b0
// which holds the four products in 10-bit fields f = r + 2 t (row r, column 2p + t). The pre-adder
// forms b1 2^20 + b0; a1 2^10 + a0 is shared by the N / 2 column pairs. A cascade of two slices
// adds elements 2i and 2i + 1 and a bias of 512 per field, so each field holds a pair sum + 512 in
// [224, 800] and never borrows from the next. The 8 fields of a block add to sigma + 4096 in
// [0, 8192): flipping bit 12 gives sigma. DSP slices: 32 N for the products, 8 N in fp4_fedp_acc.
module fp4_fedp_dsp
    import dotprod_pkg::*;
#(
    parameter int N = 2  // columns, even
) (
    input  logic         clk,
    input  logic         rst,
    input  logic         in_vld,
    input  logic [255:0] a [2],     // element k of row r is a[r][4k+3:4k]
    input  logic [255:0] b [N],
    input  logic [1:0]   mode,
    input  logic [31:0]  sfa [2],
    input  logic [31:0]  sfb [N],
    input  logic [31:0]  c [2][N],
    output logic         out_vld,
    output logic [31:0]  d [2][N]
);
    localparam logic signed [47:0] BIAS = 48'h80_2008_0200;  // 512 in each field

    initial begin
        if (N < 2 || N % 2 != 0) $fatal(1, "fp4_fedp_dsp: N must be even");
    end

    // E2M1 code -> signed integer in units of 1/2
    function automatic logic signed [4:0] e2m1(input logic [3:0] x);
        logic signed [4:0] m;
        case (x[2:0])
            3'd0: m = 5'sd0;
            3'd1: m = 5'sd1;
            3'd2: m = 5'sd2;
            3'd3: m = 5'sd3;
            3'd4: m = 5'sd4;
            3'd5: m = 5'sd6;
            3'd6: m = 5'sd8;
            default: m = 5'sd12;
        endcase
        return x[3] ? -m : m;
    endfunction

    // ---------------------------------------------------------------- stage 1: products in DSP slices
    logic signed [17:0] qa [64];  // multiplier operand of element k: a1 2^10 + a0
    for (genvar k = 0; k < 64; k++) begin : g_qa
        assign qa[k] = (18'(e2m1(a[1][4*k +: 4])) <<< 10) + 18'(e2m1(a[0][4*k +: 4]));
    end

    logic [39:0] s1_f [N/2][32];  // fields of column pair p, elements 2i and 2i + 1
    for (genvar p = 0; p < N / 2; p++) begin : g_pair
        logic signed [42:0] m [64];
        for (genvar k = 0; k < 64; k++) begin : g_k
            wire signed [24:0] pb = (25'(e2m1(b[2*p+1][4*k +: 4])) <<< 20) + 25'(e2m1(b[2*p][4*k +: 4]));
            assign m[k] = pb * qa[k];
        end
        always_ff @(posedge clk)
            for (int i = 0; i < 32; i++) s1_f[p][i] <= 40'(48'(m[2*i]) + BIAS + 48'(m[2*i+1]));
    end

    // If all 64 products of an output are zero, d = C (+0 for -0)
    logic [63:0] nz_a [2], nz_b [N];
    for (genvar k = 0; k < 64; k++) begin : g_nz
        for (genvar r = 0; r < 2; r++) begin : g_a
            assign nz_a[r][k] = a[r][4*k +: 3] != 3'd0;
        end
        for (genvar s = 0; s < N; s++) begin : g_b
            assign nz_b[s][k] = b[s][4*k +: 3] != 3'd0;
        end
    end

    // ---------------------------------------------------------------- stage 2 input: block sums
    logic signed [NVFP4_INNER_W-1:0] sigma [2][N][4];
    for (genvar p = 0; p < N / 2; p++) begin : g_sum
        for (genvar f = 0; f < 4; f++) begin : g_f
            for (genvar g = 0; g < 4; g++) begin : g_blk
                logic [12:0] u [8];
                for (genvar i = 0; i < 8; i++) begin : g_u
                    assign u[i] = 13'(s1_f[p][8*g+i][10*f +: 10]);
                end
                wire [12:0] sum = ((u[0] + u[1]) + (u[2] + u[3])) + ((u[4] + u[5]) + (u[6] + u[7]));
                assign sigma[f%2][2*p+f/2][g] = {~sum[12], sum[11:0]};
            end
        end
    end

    // ---------------------------------------------------------------- stages 2-5 per output
    logic vld [2][N];
    for (genvar r = 0; r < 2; r++) begin : g_row
        for (genvar s = 0; s < N; s++) begin : g_col
            fp4_fedp_acc u_acc (
                .clk, .rst, .in_vld, .mode, .sfa(sfa[r]), .sfb(sfb[s]), .c(c[r][s]),
                .zero((nz_a[r] & nz_b[s]) == '0), .sigma(sigma[r][s]),
                .out_vld(vld[r][s]), .d(d[r][s]));
        end
    end
    assign out_vld = vld[0][0];  // the outputs run in lockstep
endmodule
