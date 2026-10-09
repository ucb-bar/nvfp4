`timescale 1ns / 1ps
// End-to-end simulation of sophialake_top over its UART.
//   +VEC=<file>  "<n>" then n*IN_BYTES hex bytes, one per line
//   +OUT=<file>  every byte received from the FPGA, one per line
module tb_top;
    import dut_cfg_pkg::*;

    localparam real BIT_NS = 1000.0;  // 1 Mbaud

    logic       sys_clock = 1'b0;
    logic       rxd = 1'b1;
    wire        txd;
    wire  [3:0] led;

    always #10 sys_clock = ~sys_clock;

    sophialake_top dut (.sys_clock, .uart_tsi_rxd(rxd), .uart_tsi_txd(txd), .led);

    task automatic send(input logic [7:0] b);
        rxd = 1'b0;
        #(BIT_NS);
        for (int i = 0; i < 8; i++) begin
            rxd = b[i];
            #(BIT_NS);
        end
        rxd = 1'b1;
        #(BIT_NS);
    endtask

    int fout;
    int n_rx = 0;

    initial begin : receiver
        logic [7:0] r;
        forever begin
            @(negedge txd);
            #(BIT_NS * 1.5);
            for (int i = 0; i < 8; i++) begin
                r[i] = txd;
                #(BIT_NS);
            end
            if (txd !== 1'b1) $display("TB framing error at %t", $time);
            $fdisplay(fout, "%02x", r);
            n_rx++;
        end
    end

    initial begin
        string vf, of;
        int    fin, n;
        logic [7:0] b;

        if (!$value$plusargs("VEC=%s", vf)) $fatal(1, "missing +VEC");
        if (!$value$plusargs("OUT=%s", of)) $fatal(1, "missing +OUT");
        fin  = $fopen(vf, "r");
        fout = $fopen(of, "w");
        void'($fscanf(fin, "%d", n));

        wait (led[3]);
        #20us;
        send(8'h49);
        #(BIT_NS * 10 * 12);

        for (int v = 0; v < n; v++) begin
            send(8'hA5);
            for (int k = 0; k < IN_BYTES; k++) begin
                void'($fscanf(fin, "%h", b));
                send(b);
            end
        end
        #(BIT_NS * 10 * (OUT_BYTES + 4));
        $display("TB done: sent %0d vectors, received %0d bytes, flags led=%b", n, n_rx, led);
        $fclose(fout);
        $finish;
    end
endmodule
