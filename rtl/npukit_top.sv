// NpuKit top for PYNQ-Z2 (instantiated via npukit_pl.v).
// MVP board face: LED heartbeat on LD0; BTN0 holds reset.
// Systolic array is instantiated for hierarchy; host/AXI comes later.

module npukit_top (
    input  logic       clk,     // 125 MHz (H16)
    input  logic       btn0,    // active-high reset while pressed
    output logic [3:0] led
);

    localparam int CLK_HZ      = 125_000_000;
    localparam int BLINK_HZ    = 1;
    localparam int MAX_COUNT   = (CLK_HZ / (2 * BLINK_HZ)) - 1;
    localparam int COUNT_WIDTH = $clog2(MAX_COUNT + 1);
    localparam int N           = 8;

    logic                   rst_n;
    logic [COUNT_WIDTH-1:0] hb_count;
    logic                   led_state;

    logic signed [7:0]  a_west  [0:N-1];
    logic signed [7:0]  b_north [0:N-1];
    wire  signed [31:0] c_out   [0:N*N-1];

    assign rst_n = ~btn0;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            hb_count  <= '0;
            led_state <= 1'b0;
        end else if (hb_count >= MAX_COUNT[COUNT_WIDTH-1:0]) begin
            hb_count  <= '0;
            led_state <= ~led_state;
        end else begin
            hb_count <= hb_count + 1'b1;
        end
    end

    genvar gi;
    generate
        for (gi = 0; gi < N; gi++) begin : gen_tie
            assign a_west[gi]  = 8'sd0;
            assign b_north[gi] = 8'sd0;
        end
    endgenerate

    systolic_array #(
        .N(N)
    ) u_array (
        .clk    (clk),
        .rst_n  (rst_n),
        .clear  (1'b0),
        .enable (1'b0),
        .a_west (a_west),
        .b_north(b_north),
        .c_out  (c_out)
    );

    assign led[0]   = led_state;
    assign led[3:1] = 3'b000;

endmodule
