// Weight-stationary NVFP4 tensor core: PM x PN fp4_fedp units, each a 64-element dot product with
// NVIDIA tensor-core arithmetic. Default M x K x N per step = 2 x 64 x 4 (512 MACs per cycle).
// DSP = 1: each pair of rows is one fp4_fedp_dsp, with the E2M1 products in 32 x PN DSP48E1 slices.
// DSP = 0: fp4_fedp units, products in LUTs. Both give the same outputs.
// Schedule: for jb, for kc, load B[jb,kc], for ib, C[ib,jb] += A[ib,kc] * B[jb,kc].
// B RAM outputs hold the weights throughout the innermost row-block reuse loop.
// Drain before advancing K so every accumulator entry observes its prior write.
// C is a banked multi-entry read/write accumulator, one entry per output block.
// K may be a multiple of 32: the last chunk of a K = 64 c + 32 tile zeroes the elements of A past K
// and both scales past K, so it equals an MMA on operands zero-padded to 64 with zero scales.
//
// Register map (byte offsets in the 64 KiB device):
//   0x0000  ID      RO  0x4E56_5443 ("NVTC")
//   0x0004  TILE    RO  maximum tile {M[31:24], N[23:16], K[15:0]} = {64, 64, 256}
//   0x0008  ARRAY   RO  {PM[15:8], PN[7:0]}
//   0x000C  CTRL    WO  [0] START: C = A x B + C; with [1] OVERWRITE: C = A x B
//   0x0010  STATUS  RO  [0] busy, [1] the last START was rejected (DIMS not valid)
//   0x0014  MODE    RW  [1:0] 0 NVFP4, 1 MXFP4 block 32, 2 FP4, 3 MXFP4 block 16
//   0x0018  DIMS    RW  tile of the next START {M[31:24], N[23:16], K[15:0]}; M a multiple of PM,
//                       N a multiple of PN, K a multiple of 32. Reset value: the maximum tile.
//   0x001C  OPS     RO  completed STARTs
//   0x0020  CYCLES  RO  cycles of the last START
//   0x0024  PK      RO  physical reduction width, 64
//
// Memory map. Rows of A and columns of B are 128 B: element k is nibble k, low nibble first.
// Scale rows are 16 B: byte g scales elements [16g, 16g + 16) (NVFP4). Writes must be whole
// words; a partial write is denied.
//   0x2000  A      64 rows x 128 B,     row i at 0x2000 + 128 i
//   0x4000  SFA    64 rows x 16 B,      row i at 0x4000 + 16 i
//   0x6000  B      64 columns x 128 B,  column j at 0x6000 + 128 j
//   0x8000  SFB    64 columns x 16 B,   column j at 0x8000 + 16 j
//   0xC000  C      64 x 64 FP32, row-major, C[i][j] at 0xC000 + 256 i + 4 j
//
// While the core runs, only register reads are served; other accesses wait.
// A bus request carries any words of one 32-byte beat. A beat of A or B elements takes one cycle;
// other words take one cycle each.
module nvfp4_tc
    import tl_pkg::*;
#(
    parameter int PM = 2,  // rows of A per step, a power of 2, at least 2
    parameter int PN = 4,  // columns of B per step, a power of 2, at least 2
    parameter bit DSP = 1  // products in DSP slices (fp4_fedp_dsp) or in LUTs (fp4_fedp)
) (
    input  logic clk,
    input  logic rst,
    input  logic a_valid,
    output logic a_ready,
    input  wa_t  a,
    output logic d_valid,
    input  logic d_ready,
    output wd_t  d,
    output logic busy
);
    localparam int TM  = 64;
    localparam int TN  = 64;
    localparam int TK  = 256;
    localparam int PK  = 64;       // reduction elements per step
    localparam int KC  = TK / PK;  // K chunks
    localparam int MB  = TM / PM;  // blocks of C along M
    localparam int NB  = TN / PN;  // blocks of C along N
    localparam int EW  = PK / 8;      // element words per K chunk
    localparam int NL  = EW + 1;      // element words plus one scale word
    localparam int LK = $clog2(PK);
    localparam int LPM = $clog2(PM);
    localparam int LPN = $clog2(PN);
    localparam int LKC = $clog2(KC);
    localparam int LMB = $clog2(MB);
    localparam int LNB = $clog2(NB);

    initial begin
        if (PM < 2 || PM > 32 || (PM & (PM - 1)) != 0 ||
            PN < 2 || PN > 32 || (PN & (PN - 1)) != 0)
            $fatal(1, "PM/PN must be powers of two in [2,32]");
    end

    // ---------------------------------------------------------------- bus front end
    // A request covers words of one 32-byte beat. It is latched, then served: the words of an A
    // or B element beat sit in separate RAM lanes and take one cycle together; other words take
    // one cycle each, lowest first. Read data arrives one cycle after its word is served.
    typedef enum logic [2:0] {R_REG, R_AE, R_AS, R_BE, R_BS, R_C, R_NONE} region_e;

    function automatic region_e region_of(logic [15:0] o);
        if      (o < 16'h0100)          return R_REG;
        else if (o[15:13] == 3'b001)    return R_AE;  // 0x2000 - 0x3FFF
        else if (o[15:10] == 6'b010000) return R_AS;  // 0x4000 - 0x43FF
        else if (o[15:13] == 3'b011)    return R_BE;  // 0x6000 - 0x7FFF
        else if (o[15:10] == 6'b100000) return R_BS;  // 0x8000 - 0x83FF
        else if (o[15:14] == 2'b11)     return R_C;   // 0xC000 - 0xFFFF
        else                            return R_NONE;
    endfunction

    wa_t        req;
    logic       active;  // a request is latched
    logic [7:0] todo;    // its words still to serve
    logic [2:0] w;       // lowest word still to serve
    always_comb begin
        w = '0;
        for (int i = 7; i >= 0; i--) if (todo[i]) w = 3'(i);
    end

    wire [15:0] off    = {req.address[15:5], w, 2'b00};
    wire        wr     = req.opcode != A_GET;
    region_e    region;
    assign region = region_of(off);
    wire        lanes  = region == R_AE || region == R_BE;
    wire [7:0]  served = lanes ? todo : 8'(1) << w;
    wire        serve  = active && todo != 0;
    wire [31:0] wdata  = req.data[32*w +: 32];
    logic       whole;  // every served word is written whole
    always_comb begin
        whole = 1'b1;
        for (int i = 0; i < 8; i++) if (served[i] && req.mask[4*i +: 4] != 4'hF) whole = 1'b0;
    end

    // Host location in the banked memories
    logic [5:0]     hrow;  // row of A or column of B
    logic [LKC-1:0] hkc;   // K chunk
    always_comb begin
        if (lanes) begin
            hrow = off[12:7];
            hkc  = off[6:LK-1];
        end else begin
            hrow = off[9:4];
            hkc  = off[3:2];
        end
    end
    wire [5:0] hci = off[13:8];  // C row
    wire [5:0] hcj = off[7:2];   // C column

    wire [LPM-1:0]       ha_bank = hrow[LPM-1:0];
    wire [LMB+LKC-1:0]   ha_addr = {hrow[5:LPM], hkc};
    wire [LPN-1:0]       hb_bank = hrow[LPN-1:0];
    wire [LNB+LKC-1:0]   hb_addr = {hrow[5:LPN], hkc};
    wire [LPM+LPN-1:0]   hc_lane = {hci[LPM-1:0], hcj[LPN-1:0]};
    wire [LMB+LNB-1:0]   hc_addr = {hci[5:LPM], hcj[5:LPN]};

    wire is_mem = region != R_REG && region != R_NONE;

    // While the core runs, only register reads are accepted.
    assign a_ready = !active && !d_valid &&
                     !(busy && (a.opcode != A_GET || region_of(a.address[15:0]) != R_REG));
    wire mem_wr = serve && wr && is_mem && whole;
    wire mem_rd = serve && !wr && is_mem;

    // ---------------------------------------------------------------- registers
    logic [1:0]  mode;
    logic [7:0]  dim_m, dim_n;
    logic [15:0] dim_k;
    logic [31:0] ops, cycles, last_cycles, reg_rdata;
    logic        rejected;

    always_comb begin
        case (off[7:2])
            6'd0:    reg_rdata = 32'h4E56_5443;
            6'd1:    reg_rdata = {8'(TM), 8'(TN), 16'(TK)};
            6'd2:    reg_rdata = {16'd0, 8'(PM), 8'(PN)};
            6'd4:    reg_rdata = {30'd0, rejected, busy};
            6'd5:    reg_rdata = {30'd0, mode};
            6'd6:    reg_rdata = {dim_m, dim_n, dim_k};
            6'd7:    reg_rdata = ops;
            6'd8:    reg_rdata = last_cycles;
            6'd9:    reg_rdata = 32'(PK);
            default: reg_rdata = '0;
        endcase
    end

    wire reg_wr = serve && wr && region == R_REG && whole && !busy;
    wire start  = reg_wr && off[7:2] == 6'd3 && wdata[0];
    wire dims_ok = dim_m != 0 && dim_m <= 8'(TM) && dim_m[LPM-1:0] == 0
                && dim_n != 0 && dim_n <= 8'(TN) && dim_n[LPN-1:0] == 0
                && dim_k != 0 && dim_k <= 16'(TK) && dim_k[4:0] == 0;

    // ---------------------------------------------------------------- bus response
    logic               cap;  // read data of the words served last cycle is at the RAM outputs
    logic [7:0]         cap_words;
    region_e            sel_region;
    logic [LPM-1:0]     sel_abank;
    logic [LPN-1:0]     sel_bbank;
    logic [LPM+LPN-1:0] sel_clane;
    logic [2:0]         sel_w;
    logic [31:0]        mem_rdata;
    logic [31:0]        lane_rdata [8];

    always_ff @(posedge clk) begin
        if (rst) begin
            active  <= 1'b0;
            d_valid <= 1'b0;
            cap     <= 1'b0;
        end else begin
            if (d_valid && d_ready) d_valid <= 1'b0;
            if (a_valid && a_ready) begin
                req    <= a;
                todo   <= words_of(a.mask);
                active <= 1'b1;
                d      <= '{opcode: a.opcode == A_GET ? D_ACK_DATA : D_ACK, denied: a.mask == 0,
                            data: '0};
            end
            cap <= mem_rd;
            if (serve) begin
                todo       <= todo & ~served;
                cap_words  <= served;
                sel_region <= region;
                sel_abank  <= ha_bank;
                sel_bbank  <= hb_bank;
                sel_clane  <= hc_lane;
                sel_w      <= w;
                if (region == R_NONE || (wr && (!whole || (region == R_REG && busy))))
                    d.denied <= 1'b1;
                if (!wr && region == R_REG) d.data[32*w +: 32] <= reg_rdata;
            end
            if (cap) begin
                if (sel_region == R_AE || sel_region == R_BE) begin
                    for (int i = 0; i < 8; i++) if (cap_words[i]) d.data[32*i +: 32] <= lane_rdata[i];
                end else begin
                    d.data[32*sel_w +: 32] <= mem_rdata;
                end
            end
            if (active && todo == 0) begin  // the last capture, if any, is in this cycle
                active  <= 1'b0;
                d_valid <= 1'b1;
            end
        end
    end

    // ---------------------------------------------------------------- sequencer
    typedef enum logic [1:0] {S_IDLE, S_LOAD, S_ISSUE, S_DRAIN} seq_e;

    seq_e                seq;
    logic [LMB-1:0]      ib, ib_last;
    logic [LNB-1:0]      jb, jb_last;
    logic [LKC-1:0]      kc, kc_last;
    logic                overwrite;
    logic                k_tail;   // K = 64 c + 32: the last chunk holds 32 elements
    logic                mask_hi;  // the current chunk is that last chunk
    logic [LMB+LNB:0]    inflight;

    wire issue = seq == S_ISSUE;
    logic op_vld, op_zero, res_vld;

    assign busy = seq != S_IDLE;

    always_ff @(posedge clk) begin
        if (rst) begin
            seq         <= S_IDLE;
            mode        <= '0;
            dim_m       <= 8'(TM);
            dim_n       <= 8'(TN);
            dim_k       <= 16'(TK);
            ops         <= '0;
            last_cycles <= '0;
            rejected    <= 1'b0;
            inflight    <= '0;
        end else begin
            inflight <= inflight + (LMB + LNB + 1)'(issue) - (LMB + LNB + 1)'(res_vld);
            if (reg_wr && off[7:2] == 6'd5) mode <= wdata[1:0];
            if (reg_wr && off[7:2] == 6'd6) {dim_m, dim_n, dim_k} <= wdata;
            case (seq)
                S_IDLE: if (start) begin
                    rejected <= !dims_ok;
                    if (dims_ok) begin
                        seq       <= S_LOAD;
                        overwrite <= wdata[1];
                        ib        <= '0;
                        jb        <= '0;
                        kc        <= '0;
                        ib_last   <= LMB'(dim_m[7:LPM] - 1'b1);
                        jb_last   <= LNB'(dim_n[7:LPN] - 1'b1);
                        kc_last   <= LKC'((dim_k - 1'b1) >> LK);
                        k_tail    <= dim_k[5];
                        cycles    <= '0;
                    end
                end
                S_LOAD: begin
                    cycles <= cycles + 1'b1;
                    seq <= S_ISSUE;
                end
                S_ISSUE: begin
                    cycles <= cycles + 1'b1;
                    if (ib == ib_last) seq <= S_DRAIN;
                    else ib <= ib + 1'b1;
                end
                S_DRAIN: begin
                    cycles <= cycles + 1'b1;
                    if (inflight == 0) begin
                        ib <= '0;
                        if (kc != kc_last) begin
                            kc <= kc + 1'b1;
                            seq <= S_LOAD;
                        end else if (jb != jb_last) begin
                            kc <= '0;
                            jb <= jb + 1'b1;
                            seq <= S_LOAD;
                        end else begin
                            seq         <= S_IDLE;
                            ops         <= ops + 1'b1;
                            last_cycles <= cycles + 1'b1;
                        end
                    end
                end
                default: seq <= S_IDLE;
            endcase
        end
    end

    // The RAM outputs of an issue are valid one cycle later, together with op_vld.
    logic [LMB+LNB-1:0] op_caddr, wb_addr;
    always_ff @(posedge clk) begin
        op_vld   <= !rst && issue;
        op_zero  <= overwrite && kc == 0;
        op_caddr <= {ib, jb};
        if (seq == S_LOAD) mask_hi <= k_tail && kc == kc_last;
    end

    // C block address of each operation in flight, in issue order
    sync_fifo #(.W(LMB + LNB), .DEPTH(16)) u_wb (
        .clk, .rst,
        .wdata(op_caddr), .wvalid(op_vld), .wready(),
        .rdata(wb_addr), .rvalid(), .rready(res_vld));

    // ---------------------------------------------------------------- memories
    logic [31:0] a_dout [PM][NL];
    logic [31:0] b_dout [PN][NL];
    logic [31:0] c_dout [PM][PN];
    logic [31:0] res    [PM][PN];

    for (genvar r = 0; r < PM; r++) begin : g_a
        for (genvar l = 0; l < EW; l++) begin : g_lane
            wire hsel = region == R_AE && ha_bank == LPM'(r) && served[l];
            sdp_ram #(.DEPTH(MB * KC)) u_ram (
                .clk,
                .we(mem_wr && hsel), .waddr(ha_addr), .wdata(req.data[32*l +: 32]),
                .re(busy ? issue : mem_rd && hsel), .raddr(busy ? {ib, kc} : ha_addr),
                .rdata(a_dout[r][l]));
        end
        wire scale_sel = region == R_AS && ha_bank == LPM'(r);
        sdp_ram #(.DEPTH(MB * 4)) u_scale (
            .clk, .we(mem_wr && scale_sel),
            .waddr({hrow[5:LPM], off[3:2]}), .wdata,
            .re(busy ? issue : mem_rd && scale_sel),
            .raddr(busy ? {ib, kc} : {hrow[5:LPM], off[3:2]}),
            .rdata(a_dout[r][EW]));
    end

    for (genvar s = 0; s < PN; s++) begin : g_b
        for (genvar l = 0; l < EW; l++) begin : g_lane
            wire hsel = region == R_BE && hb_bank == LPN'(s) && served[l];
            sdp_ram #(.DEPTH(NB * KC)) u_ram (
                .clk,
                .we(mem_wr && hsel), .waddr(hb_addr), .wdata(req.data[32*l +: 32]),
                .re(busy ? seq == S_LOAD : mem_rd && hsel), .raddr(busy ? {jb, kc} : hb_addr),
                .rdata(b_dout[s][l]));
        end
        wire scale_sel = region == R_BS && hb_bank == LPN'(s);
        sdp_ram #(.DEPTH(NB * 4)) u_scale (
            .clk, .we(mem_wr && scale_sel),
            .waddr({hrow[5:LPN], off[3:2]}), .wdata,
            .re(busy ? seq == S_LOAD : mem_rd && scale_sel),
            .raddr(busy ? {jb, kc} : {hrow[5:LPN], off[3:2]}),
            .rdata(b_dout[s][EW]));
    end

    for (genvar r = 0; r < PM; r++) begin : g_c
        for (genvar s = 0; s < PN; s++) begin : g_lane
            wire hsel = region == R_C && hc_lane == (LPM + LPN)'(r * PN + s);
            sdp_ram #(.DEPTH(MB * NB)) u_ram (
                .clk,
                .we(busy ? res_vld : mem_wr && hsel), .waddr(busy ? wb_addr : hc_addr),
                .wdata(busy ? res[r][s] : wdata),
                .re(busy ? issue : mem_rd && hsel), .raddr(busy ? {ib, jb} : hc_addr),
                .rdata(c_dout[r][s]));
        end
    end

    always_comb begin
        for (int i = 0; i < 8; i++)
            lane_rdata[i] = sel_region == R_AE ? a_dout[sel_abank][i] : b_dout[sel_bbank][i];
        case (sel_region)  // element beats return through lane_rdata
            R_AS:    mem_rdata = a_dout[sel_abank][EW];
            R_BS:    mem_rdata = b_dout[sel_bbank][EW];
            R_C:     mem_rdata = c_dout[sel_clane[LPM+LPN-1:LPN]][sel_clane[LPN-1:0]];
            default: mem_rdata = '0;
        endcase
    end

    // ---------------------------------------------------------------- dot-product grid
    // Past K, zero the elements of A (so every product is zero) and the scales of A and B (so the
    // padding blocks, at exponent -252, cannot raise the alignment exponent). Mode 1 keeps byte 0.
    wire [31:0] sf_keep = !mask_hi ? '1 : mode == 2'd1 ? 32'h0000_00FF : 32'h0000_FFFF;
    logic [PK*4-1:0] a_vec [PM], b_vec [PN];
    logic [31:0]     sfa [PM], sfb [PN], c_in [PM][PN];

    for (genvar r = 0; r < PM; r++) begin : g_a_in
        for (genvar l = 0; l < EW; l++) begin : g_w
            assign a_vec[r][32*l +: 32] = mask_hi && l >= EW / 2 ? '0 : a_dout[r][l];
        end
        assign sfa[r] = a_dout[r][EW] & sf_keep;
        for (genvar s = 0; s < PN; s++) begin : g_c
            assign c_in[r][s] = op_zero ? '0 : c_dout[r][s];
        end
    end
    for (genvar s = 0; s < PN; s++) begin : g_b_in
        for (genvar l = 0; l < EW; l++) begin : g_w
            assign b_vec[s][32*l +: 32] = b_dout[s][l];
        end
        assign sfb[s] = b_dout[s][EW] & sf_keep;
    end

    logic [PM/2-1:0] out_vld;
    for (genvar p = 0; p < PM / 2; p++) begin : g_rows
        if (DSP) begin : g_dsp
            logic [31:0] pair_res [2][PN];
            fp4_fedp_dsp #(.N(PN)) u_dot (
                .clk, .rst, .in_vld(op_vld), .a('{a_vec[2*p], a_vec[2*p+1]}), .b(b_vec), .mode,
                .sfa('{sfa[2*p], sfa[2*p+1]}), .sfb, .c('{c_in[2*p], c_in[2*p+1]}),
                .out_vld(out_vld[p]), .d(pair_res));
            assign res[2*p]   = pair_res[0];
            assign res[2*p+1] = pair_res[1];
        end else begin : g_lut
            logic [2*PN-1:0] vld;
            for (genvar t = 0; t < 2; t++) begin : g_row
                for (genvar s = 0; s < PN; s++) begin : g_col
                    fp4_fedp u_dot (
                        .clk, .rst, .in_vld(op_vld), .a(a_vec[2*p+t]), .b(b_vec[s]), .mode,
                        .sfa(sfa[2*p+t]), .sfb(sfb[s]), .c(c_in[2*p+t][s]),
                        .out_vld(vld[t*PN+s]), .d(res[2*p+t][s]));
                end
            end
            assign out_vld[p] = vld[0];
        end
    end

    assign res_vld = out_vld[0];  // the units run in lockstep
endmodule
