// NpuKit top for PYNQ-Z2 (instantiated via npukit_pl.v).
// Clocked from PS FCLK0 (100 MHz). AXI4-Lite programs A/B and reads C.
// Board face: LD0 heartbeat, LD1 busy, LD2 done; BTN0 holds reset.

module npukit_top (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        btn0,

    input  logic [15:0] s_axi_awaddr,
    input  logic        s_axi_awvalid,
    output logic        s_axi_awready,
    input  logic [31:0] s_axi_wdata,
    input  logic [3:0]  s_axi_wstrb,
    input  logic        s_axi_wvalid,
    output logic        s_axi_wready,
    output logic [1:0]  s_axi_bresp,
    output logic        s_axi_bvalid,
    input  logic        s_axi_bready,

    input  logic [15:0] s_axi_araddr,
    input  logic        s_axi_arvalid,
    output logic        s_axi_arready,
    output logic [31:0] s_axi_rdata,
    output logic [1:0]  s_axi_rresp,
    output logic        s_axi_rvalid,
    input  logic        s_axi_rready,

    input  logic [31:0] s_axis_tdata,
    input  logic        s_axis_tvalid,
    output logic        s_axis_tready,
    input  logic        s_axis_tlast,

    output logic [31:0] m_axis_tdata,
    output logic        m_axis_tvalid,
    input  logic        m_axis_tready,
    output logic        m_axis_tlast,

    output logic [3:0]  led
);

    localparam int CLK_HZ      = 100_000_000;
    localparam int BLINK_HZ    = 1;
    localparam int MAX_COUNT   = (CLK_HZ / (2 * BLINK_HZ)) - 1;
    localparam int COUNT_WIDTH = $clog2(MAX_COUNT + 1);

    logic                   rst_n_i;
    logic [COUNT_WIDTH-1:0] hb_count;
    logic                   led_state;
    logic                   busy;
    logic                   done;

    assign rst_n_i = rst_n & ~btn0;

    always_ff @(posedge clk or negedge rst_n_i) begin
        if (!rst_n_i) begin
            hb_count  <= '0;
            led_state <= 1'b0;
        end else if (hb_count >= MAX_COUNT[COUNT_WIDTH-1:0]) begin
            hb_count  <= '0;
            led_state <= ~led_state;
        end else begin
            hb_count <= hb_count + 1'b1;
        end
    end

    npukit_axil #(
        .N(8)
    ) u_axil (
        .clk          (clk),
        .rst_n        (rst_n_i),
        .s_axi_awaddr (s_axi_awaddr),
        .s_axi_awvalid(s_axi_awvalid),
        .s_axi_awready(s_axi_awready),
        .s_axi_wdata  (s_axi_wdata),
        .s_axi_wstrb  (s_axi_wstrb),
        .s_axi_wvalid (s_axi_wvalid),
        .s_axi_wready (s_axi_wready),
        .s_axi_bresp  (s_axi_bresp),
        .s_axi_bvalid (s_axi_bvalid),
        .s_axi_bready (s_axi_bready),
        .s_axi_araddr (s_axi_araddr),
        .s_axi_arvalid(s_axi_arvalid),
        .s_axi_arready(s_axi_arready),
        .s_axi_rdata  (s_axi_rdata),
        .s_axi_rresp  (s_axi_rresp),
        .s_axi_rvalid (s_axi_rvalid),
        .s_axi_rready (s_axi_rready),
        .s_axis_tdata (s_axis_tdata),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready),
        .s_axis_tlast (s_axis_tlast),
        .m_axis_tdata (m_axis_tdata),
        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready),
        .m_axis_tlast (m_axis_tlast),
        .busy         (busy),
        .done         (done)
    );

    assign led[0] = led_state;
    assign led[1] = busy;
    assign led[2] = done;
    assign led[3] = 1'b0;

endmodule
