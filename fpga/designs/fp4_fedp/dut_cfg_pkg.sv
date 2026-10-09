package dut_cfg_pkg;
    // a[255:0], b[255:0], sfa[31:0], sfb[31:0], c[31:0], mode[7:0] (same frame as allegro_fp4)
    localparam int          IN_BYTES  = 77;
    // d (FP32)
    localparam int          OUT_BYTES = 4;
    localparam logic [31:0] DESIGN_ID = "FED4";
endpackage
