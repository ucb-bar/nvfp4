package dut_cfg_pkg;
    // a_row[2][32], b_col[2][32] (8 E2M1 per word), sf_a, sf_b (UE4M3), c_val (FP32), fmt_s
    localparam int          IN_BYTES  = 23;
    // d_val (FP32)
    localparam int          OUT_BYTES = 4;
    localparam logic [31:0] DESIGN_ID = "VXF4";
endpackage
