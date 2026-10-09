`timescale 1ns / 1ps
// Drives nvfp4_tc's TL-UL port directly, one word per request through tl_widen, to check what the
// UART host is too slow to reach: accesses while the core runs, dependent K chunks on one C block,
// and denied partial writes.
module tb_tc;
    import tl_pkg::*;

    logic clk = 1'b0, rst = 1'b1;
    logic a_valid = 1'b0, a_ready, d_valid, d_ready = 1'b1, busy;
    a_t   a;
    d_t   d;
    int   fails = 0;

    always #10 clk = ~clk;

    logic w_a_valid, w_a_ready, w_d_valid, w_d_ready;
    wa_t  w_a;
    wd_t  w_d;

    tl_widen u_widen (.clk, .a_valid, .a_ready, .a, .d_valid, .d_ready, .d,
                      .w_a_valid, .w_a_ready, .w_a, .w_d_valid, .w_d_ready, .w_d);
    nvfp4_tc dut (.clk, .rst, .a_valid(w_a_valid), .a_ready(w_a_ready), .a(w_a),
                  .d_valid(w_d_valid), .d_ready(w_d_ready), .d(w_d), .busy);

    // One TL access. Returns the read data, the denied bit and the cycles spent waiting for a_ready.
    task automatic access(input a_op_e op, input logic [31:0] addr, input logic [31:0] wdata,
                          input logic [3:0] mask, output logic [31:0] rdata, output logic denied,
                          output int stalled);
        @(negedge clk);
        a       = '{opcode: op, address: addr, mask: mask, data: wdata};
        a_valid = 1'b1;
        stalled = 0;
        #1;
        while (!a_ready) begin
            stalled++;
            if (stalled > 10000) $fatal(1, "request to 0x%04x never accepted", addr);
            @(negedge clk);
            #1;
        end
        @(posedge clk);
        #1 a_valid = 1'b0;
        while (!d_valid) begin
            @(posedge clk);
            #1;
        end
        rdata  = d.data;
        denied = d.denied;
    endtask

    task automatic wr(input logic [31:0] addr, input logic [31:0] data);
        logic [31:0] r;
        logic dn;
        int s;
        access(A_PUT_FULL, addr, data, 4'hF, r, dn, s);
    endtask

    task automatic rd(input logic [31:0] addr, output logic [31:0] r, output int s);
        logic dn;
        access(A_GET, addr, '0, 4'hF, r, dn, s);
    endtask

    task automatic expect_eq(input string what, input logic [31:0] got, input logic [31:0] want);
        if (got !== want) begin
            fails++;
            $display("FAIL %s: got 0x%08x want 0x%08x", what, got, want);
        end else begin
            $display("ok   %s = 0x%08x", what, got);
        end
    endtask

    initial begin
        logic [31:0] r;
        logic dn;
        int s;
        repeat (5) @(posedge clk);
        rst = 1'b0;

        // Rows 0-1 of A and columns 0-3 of B: 256 x 1.0 with unit scales.
        for (int i = 0; i < 2; i++) begin
            for (int w = 0; w < 32; w++) wr(32'h2000 + 128 * i + 4 * w, 32'h2222_2222);
            for (int q = 0; q < 4; q++)  wr(32'h4000 + 16 * i + 4 * q, 32'h3838_3838);
        end
        for (int j = 0; j < 8; j++) begin
            for (int w = 0; w < 32; w++) wr(32'h6000 + 128 * j + 4 * w, 32'h2222_2222);
            for (int q = 0; q < 4; q++)  wr(32'h8000 + 16 * j + 4 * q, 32'h3838_3838);
        end

        // One 2 x 8 block of C: the eight K chunks update the same C entries back to back.
        wr(32'h0018, {8'd2, 8'd8, 16'd256});
        wr(32'h000C, 32'h3);  // START, OVERWRITE
        rd(32'h0010, r, s);
        expect_eq("STATUS right after START (served while busy)", r, 32'h1);
        rd(32'hC000 + 256 * 1 + 4 * 3, r, s);
        expect_eq("C[1][3] = 256 x 1.0, read issued while busy", r, 32'h4380_0000);
        if (s < 4) begin fails++; $display("FAIL C read waited only %0d cycles", s); end
        else $display("ok   C read waited %0d cycles for the core", s);
        rd(32'hC000, r, s);
        expect_eq("C[0][0]", r, 32'h4380_0000);

        wr(32'h0018, {8'd2, 8'd8, 16'd64});
        wr(32'h000C, 32'h1);  // START, accumulate: 256 + 64
        access(A_PUT_FULL, 32'hC004, 32'h3F80_0000, 4'hF, r, dn, s);
        if (s < 4) begin fails++; $display("FAIL C write waited only %0d cycles", s); end
        rd(32'hC000, r, s);
        expect_eq("C[0][0] = 256 + 64", r, 32'h43A0_0000);
        rd(32'hC004, r, s);
        expect_eq("C[0][1] written after the run", r, 32'h3F80_0000);

        access(A_PUT_PARTIAL, 32'h2000, 32'h0, 4'h3, r, dn, s);
        expect_eq("partial write denied", 32'(dn), 32'h1);
        rd(32'h2000, r, s);
        expect_eq("A unchanged by the denied write", r, 32'h2222_2222);

        wr(32'h0018, {8'd3, 8'd8, 16'd64});
        wr(32'h000C, 32'h1);
        rd(32'h0010, r, s);
        expect_eq("STATUS after START with DIMS 3x8x64", r, 32'h2);
        rd(32'h001C, r, s);
        expect_eq("OPS", r, 32'd2);
        rd(32'h0100, r, s);
        expect_eq("unmapped offset reads 0", r, 32'h0);

        if (fails == 0) $display("PASS");
        else $display("FAIL (%0d)", fails);
        $finish;
    end
endmodule
