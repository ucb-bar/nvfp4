package dut_cfg_pkg;
    // a[0:4] and b[0:4], 16 bits each: entries 0-3 hold 16 E2M1 nibbles, entry 4 the UE4M3 scale in bits 7:0.
    localparam int          IN_BYTES  = 20;
    // FP32 result, then status {sat, invalid, is_nan, is_inf}.
    localparam int          OUT_BYTES = 5;
    localparam logic [31:0] DESIGN_ID = "DPF4";
endpackage
