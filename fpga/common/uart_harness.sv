// Framed UART test harness around one DUT wrapper.
//
// Host -> FPGA
//   0xA5 <IN_BYTES bytes, little-endian>   run one test vector
//   0x49                                   identify
// FPGA -> host
//   0x5A <OUT_BYTES bytes, little-endian> <latency>   result of one vector
//   0x49 <id:4> <in_bytes:2> <out_bytes:2> <clk_mhz:1> <flags:1>
//
// latency is the number of cycles from start to done, saturated at 255.
// flags[0]: a result was dropped because the transmitter was busy.
// flags[1]: a frame was abandoned after an inter-byte timeout.
module uart_harness #(
    parameter int          CLK_HZ    = 50_000_000,
    parameter int          BAUD      = 1_000_000,
    parameter int          IN_BYTES  = 4,
    parameter int          OUT_BYTES = 4,
    parameter logic [31:0] DESIGN_ID = 32'h0
) (
    input  logic                   clk,
    input  logic                   rst,
    input  logic                   rxd,
    output logic                   txd,
    output logic                   start,
    output logic [IN_BYTES*8-1:0]  in_data,
    input  logic                   done,
    input  logic [OUT_BYTES*8-1:0] out_data,
    output logic [1:0]             flags,
    output logic                   activity
);
    localparam int CLKS_PER_BIT = (CLK_HZ + BAUD / 2) / BAUD;
    localparam int ID_BYTES     = 11;
    localparam int RES_BYTES    = OUT_BYTES + 2;
    localparam int TX_BYTES     = RES_BYTES > ID_BYTES ? RES_BYTES : ID_BYTES;
    localparam int TX_W         = TX_BYTES * 8;
    localparam int TIMEOUT      = CLKS_PER_BIT * 10 * 64;
    localparam int IDX_W        = $clog2(IN_BYTES + 1);
    localparam int TXN_W        = $clog2(TX_BYTES + 1);

    // Back-to-back frames are safe only if a result drains before the next frame lands.
    if (RES_BYTES > IN_BYTES + 1) $error("result frame longer than input frame");

    logic [7:0] rx_data;
    logic       rx_valid;
    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_rx (
        .clk, .rst, .rxd, .data(rx_data), .valid(rx_valid));

    logic [7:0] tx_data;
    logic       tx_valid, tx_ready;
    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_tx (
        .clk, .rst, .data(tx_data), .valid(tx_valid), .ready(tx_ready), .txd);

    // Receive side
    logic                          collecting;
    logic [IDX_W-1:0]              idx;
    logic [$clog2(TIMEOUT+1)-1:0]  idle_cnt;
    logic                          id_req;

    always_ff @(posedge clk) begin
        start  <= 1'b0;
        id_req <= 1'b0;
        if (rst) begin
            collecting <= 1'b0;
            flags[1]   <= 1'b0;
            idle_cnt   <= '0;
        end else if (rx_valid) begin
            idle_cnt <= '0;
            if (collecting) begin
                in_data <= {rx_data, in_data[IN_BYTES*8-1:8]};
                idx     <= idx + 1'b1;
                if (idx == IDX_W'(IN_BYTES - 1)) begin
                    collecting <= 1'b0;
                    start      <= 1'b1;
                end
            end else if (rx_data == 8'hA5) begin
                collecting <= 1'b1;
                idx        <= '0;
            end else if (rx_data == 8'h49) begin
                id_req <= 1'b1;
            end
        end else if (collecting) begin
            if (idle_cnt == TIMEOUT) begin
                collecting <= 1'b0;
                flags[1]   <= 1'b1;
            end else begin
                idle_cnt <= idle_cnt + 1'b1;
            end
        end
    end

    // Latency counter
    logic       busy;
    logic [7:0] lat;
    always_ff @(posedge clk) begin
        if (rst) begin
            busy <= 1'b0;
        end else if (start) begin
            busy <= 1'b1;
            lat  <= 8'd1;
        end else if (busy) begin
            if (done) busy <= 1'b0;
            else if (lat != 8'hFF) lat <= lat + 1'b1;
        end
    end

    // Transmit side
    logic [TX_W-1:0]       tx_buf;
    logic [TXN_W-1:0]      tx_left;
    wire                   tx_idle = (tx_left == 0);

    assign tx_data  = tx_buf[7:0];
    assign tx_valid = !tx_idle;

    always_ff @(posedge clk) begin
        if (rst) begin
            tx_left  <= '0;
            flags[0] <= 1'b0;
            activity <= 1'b0;
        end else if (done) begin
            activity <= ~activity;
            if (tx_idle) begin
                tx_buf  <= TX_W'({busy ? lat : 8'd0, out_data, 8'h5A});
                tx_left <= TXN_W'(RES_BYTES);
            end else begin
                flags[0] <= 1'b1;
            end
        end else if (id_req && tx_idle) begin
            tx_buf  <= TX_W'({6'd0, flags, 8'(CLK_HZ / 1_000_000),
                                    16'(OUT_BYTES), 16'(IN_BYTES), DESIGN_ID, 8'h49});
            tx_left <= TXN_W'(ID_BYTES);
        end else if (tx_valid && tx_ready) begin
            tx_buf  <= tx_buf >> 8;
            tx_left <= tx_left - 1'b1;
        end
    end
endmodule
