// sashakatne/nvfp4-dotprod-formal-dv: combinational dotprod_top pinned to NVFP4 mode.
// Inputs and outputs are registered, so the latency is 2 cycles.
module dut_wrap
    import dut_cfg_pkg::*;
    import dotprod_pkg::*;
(
    input  logic                   clk,
    input  logic                   rst,
    input  logic                   start,
    input  logic [IN_BYTES*8-1:0]  in_data,
    output logic                   done,
    output logic [OUT_BYTES*8-1:0] out_data
);
    logic [BF16_W-1:0] a [N_LANES];
    logic [BF16_W-1:0] b [N_LANES];
    logic              issued;

    always_ff @(posedge clk) begin
        if (start) begin
            for (int i = 0; i < N_LANES; i++) begin
                a[i] <= (i < 5) ? in_data[16*i +: 16]      : '0;
                b[i] <= (i < 5) ? in_data[80 + 16*i +: 16] : '0;
            end
        end
    end

    logic [FP32_W-1:0] result;
    dotprod_status_t   status;

    dotprod_top u_dotprod (
        .mode(FMT_NVFP4), .a, .b, .result, .status, .sat());

    always_ff @(posedge clk) begin
        issued <= start && !rst;
        done   <= issued;
        if (issued) out_data <= {4'd0, status, result};
    end
endmodule
