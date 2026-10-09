package dut_cfg_pkg;
    // a[255:0], b[255:0], sfa[31:0], sfb[31:0], c[31:0], mode[7:0] (same frame as fp4_fedp)
    localparam int          IN_BYTES  = 77;
    // d[0][0], d[0][1], d[1][0], d[1][1] (FP32 each)
    localparam int          OUT_BYTES = 16;
    localparam logic [31:0] DESIGN_ID = "FEDD";
endpackage
