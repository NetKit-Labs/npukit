// AXI4-Lite control + BRAM A/B/C + AXIS DMA ports + skewed feed sequencer.
// Register map (byte offsets):
//   0x000 ID       R  0x4E50554B ("NPUK")
//   0x004 VERSION  R  0x00000200
//   0x008 STATUS   R  [0] busy  [1] done  [2] axis_rx_done  [3] axis_tx_done
//   0x00C CTRL     W  [0] start  [1] clear  [2] axis_tx_arm (pulse: stream C out)
//   0x010 N_PARAM  R  N
//   0x100..0x13F   A  R/W  (also fillable via S_AXIS: 16 words A then 16 words B)
//   0x200..0x23F   B  R/W
//   0x400..0x4FF   C  R    (also readable via M_AXIS after CTRL.axis_tx_arm)

module npukit_axil #(
    parameter int N          = 8,
    parameter int ADDR_WIDTH = 16
) (
    input  logic                  clk,
    input  logic                  rst_n,

    // AXI4-Lite
    input  logic [ADDR_WIDTH-1:0] s_axi_awaddr,
    input  logic                  s_axi_awvalid,
    output logic                  s_axi_awready,
    input  logic [31:0]           s_axi_wdata,
    input  logic [3:0]            s_axi_wstrb,
    input  logic                  s_axi_wvalid,
    output logic                  s_axi_wready,
    output logic [1:0]            s_axi_bresp,
    output logic                  s_axi_bvalid,
    input  logic                  s_axi_bready,
    input  logic [ADDR_WIDTH-1:0] s_axi_araddr,
    input  logic                  s_axi_arvalid,
    output logic                  s_axi_arready,
    output logic [31:0]           s_axi_rdata,
    output logic [1:0]            s_axi_rresp,
    output logic                  s_axi_rvalid,
    input  logic                  s_axi_rready,

    // AXI-Stream slave (DMA MM2S → A then B, 32-bit words)
    input  logic [31:0]           s_axis_tdata,
    input  logic                  s_axis_tvalid,
    output logic                  s_axis_tready,
    input  logic                  s_axis_tlast,

    // AXI-Stream master (C → DMA S2MM)
    output logic [31:0]           m_axis_tdata,
    output logic                  m_axis_tvalid,
    input  logic                  m_axis_tready,
    output logic                  m_axis_tlast,

    output logic                  busy,
    output logic                  done
);

    localparam int NN       = N * N;
    localparam int FEED_CYC = 3 * N - 2;
    localparam int A_WORDS  = NN / 4;
    localparam int B_WORDS  = NN / 4;
    localparam int C_WORDS  = NN;
    localparam logic [31:0] ID_VAL      = 32'h4E50554B;
    localparam logic [31:0] VERSION_VAL = 32'h00000200;

    (* ram_style = "block" *) logic signed [7:0]  a_mem [0:NN-1];
    (* ram_style = "block" *) logic signed [7:0]  b_mem [0:NN-1];
    (* ram_style = "block" *) logic signed [31:0] c_mem [0:NN-1];

    logic               clear;
    logic               enable;
    logic signed [7:0]  a_west  [0:N-1];
    logic signed [7:0]  b_north [0:N-1];
    wire  signed [31:0] c_out   [0:NN-1];

    systolic_array #(
        .N(N)
    ) u_array (
        .clk    (clk),
        .rst_n  (rst_n),
        .clear  (clear),
        .enable (enable),
        .a_west (a_west),
        .b_north(b_north),
        .c_out  (c_out)
    );

    logic start_req;
    logic clear_req;
    logic tx_arm_req;

    // Shared by AXI-Lite write handshake and the single A/B/C memory writer.
    logic        wr_en;
    logic [11:0] wr_addr;
    logic [31:0] wr_data;
    logic [3:0]  wr_strb;
    integer      wi;
    integer      base;

    typedef enum logic [2:0] {
        ST_IDLE  = 3'd0,
        ST_CLEAR = 3'd1,
        ST_RUN   = 3'd2,
        ST_WAIT  = 3'd3,
        ST_CAPTURE = 3'd4,
        ST_DONE  = 3'd5
    } state_t;

    state_t                        state;
    logic [$clog2(FEED_CYC)-1:0]   t;
    logic                          busy_r;
    logic                          done_r;
    logic                          capture_c;
    integer                        i;

    assign busy = busy_r;
    assign done = done_r;

    // -------------------------------------------------------------------------
    // Compute sequencer: CLEAR alone, START alone (accumulate), or both
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state   <= ST_IDLE;
            t       <= '0;
            clear   <= 1'b0;
            enable  <= 1'b0;
            capture_c <= 1'b0;
            busy_r  <= 1'b0;
            done_r  <= 1'b0;
            for (i = 0; i < N; i++) begin
                a_west[i]  <= '0;
                b_north[i] <= '0;
            end
        end else begin
            clear  <= 1'b0;
            enable <= 1'b0;
            capture_c <= 1'b0;
            for (i = 0; i < N; i++) begin
                a_west[i]  <= '0;
                b_north[i] <= '0;
            end

            case (state)
                ST_IDLE: begin
                    busy_r <= 1'b0;
                    if (clear_req && start_req) begin
                        busy_r <= 1'b1;
                        done_r <= 1'b0;
                        state  <= ST_CLEAR;
                    end else if (clear_req) begin
                        busy_r <= 1'b1;
                        done_r <= 1'b0;
                        state  <= ST_CLEAR;
                    end else if (start_req) begin
                        busy_r <= 1'b1;
                        done_r <= 1'b0;
                        t      <= '0;
                        state  <= ST_RUN;
                    end
                end

                ST_CLEAR: begin
                    busy_r <= 1'b1;
                    clear  <= 1'b1;
                    t      <= '0;
                    // If start was also requested in the same CTRL write, run next
                    if (start_req || do_run_latched)
                        state <= ST_RUN;
                    else begin
                        busy_r <= 1'b0;
                        state  <= ST_IDLE;
                    end
                end

                ST_RUN: begin
                    busy_r <= 1'b1;
                    enable <= 1'b1;
                    for (i = 0; i < N; i++) begin
                        if ((t >= i[$bits(t)-1:0]) && ((t - i[$bits(t)-1:0]) < N[$bits(t)-1:0])) begin
                            a_west[i]  <= a_mem[i*N + (t - i)];
                            b_north[i] <= b_mem[(t - i)*N + i];
                        end
                    end
                    if (t == FEED_CYC[$bits(t)-1:0] - 1'b1)
                        state <= ST_WAIT;
                    else
                        t <= t + 1'b1;
                end

                ST_WAIT: begin
                    busy_r <= 1'b1;
                    capture_c <= 1'b1;
                    state  <= ST_CAPTURE;
                end

                ST_CAPTURE: begin
                    busy_r <= 1'b1;
                    state  <= ST_DONE;
                end

                ST_DONE: begin
                    done_r <= 1'b1;
                    busy_r <= 1'b0;
                    state  <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

    // Latch "run after clear" when CTRL has both bits in one write
    logic do_run_latched;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            do_run_latched <= 1'b0;
        else if (state == ST_IDLE && clear_req && start_req)
            do_run_latched <= 1'b1;
        else if (state == ST_CLEAR)
            do_run_latched <= 1'b0;
    end

    // -------------------------------------------------------------------------
    // AXIS RX: 16 words A + 16 words B
    // -------------------------------------------------------------------------
    logic [$clog2(A_WORDS+B_WORDS)-1:0] rx_idx;
    logic                               axis_rx_done;

    assign s_axis_tready = rst_n && !busy_r && (state == ST_IDLE) &&
                           (rx_idx < (A_WORDS + B_WORDS));

    // This is the only process that writes A/B/C memories.  It arbitrates
    // completion capture, AXIS payloads, and AXI-Lite payloads by priority.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rx_idx       <= '0;
            axis_rx_done <= 1'b0;
            for (wi = 0; wi < NN; wi++) begin
                a_mem[wi] <= '0;
                b_mem[wi] <= '0;
                c_mem[wi] <= '0;
            end
        end else begin
            if (clear_req)
                axis_rx_done <= 1'b0;

            if (capture_c) begin
                for (wi = 0; wi < NN; wi++)
                    c_mem[wi] <= c_out[wi];
            end

            if (s_axis_tvalid && s_axis_tready) begin
                if (rx_idx < A_WORDS) begin
                    a_mem[rx_idx*4 + 0] <= s_axis_tdata[7:0];
                    a_mem[rx_idx*4 + 1] <= s_axis_tdata[15:8];
                    a_mem[rx_idx*4 + 2] <= s_axis_tdata[23:16];
                    a_mem[rx_idx*4 + 3] <= s_axis_tdata[31:24];
                end else begin
                    b_mem[(rx_idx-A_WORDS)*4 + 0] <= s_axis_tdata[7:0];
                    b_mem[(rx_idx-A_WORDS)*4 + 1] <= s_axis_tdata[15:8];
                    b_mem[(rx_idx-A_WORDS)*4 + 2] <= s_axis_tdata[23:16];
                    b_mem[(rx_idx-A_WORDS)*4 + 3] <= s_axis_tdata[31:24];
                end
                if (rx_idx == (A_WORDS+B_WORDS)-1) begin
                    rx_idx       <= '0;
                    axis_rx_done <= s_axis_tlast;
                end else
                    rx_idx <= rx_idx + 1'b1;
            end else if (wr_en && !busy_r) begin
                if ((wr_addr >= 12'h100) && (wr_addr < 12'h140)) begin
                    base = (wr_addr - 12'h100) >> 2;
                    if (wr_strb[0]) a_mem[base*4 + 0] <= wr_data[7:0];
                    if (wr_strb[1]) a_mem[base*4 + 1] <= wr_data[15:8];
                    if (wr_strb[2]) a_mem[base*4 + 2] <= wr_data[23:16];
                    if (wr_strb[3]) a_mem[base*4 + 3] <= wr_data[31:24];
                end else if ((wr_addr >= 12'h200) && (wr_addr < 12'h240)) begin
                    base = (wr_addr - 12'h200) >> 2;
                    if (wr_strb[0]) b_mem[base*4 + 0] <= wr_data[7:0];
                    if (wr_strb[1]) b_mem[base*4 + 1] <= wr_data[15:8];
                    if (wr_strb[2]) b_mem[base*4 + 2] <= wr_data[23:16];
                    if (wr_strb[3]) b_mem[base*4 + 3] <= wr_data[31:24];
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // AXIS TX: stream C after tx_arm
    // -------------------------------------------------------------------------
    typedef enum logic [1:0] { TX_IDLE, TX_STREAM } tx_state_t;
    tx_state_t                        tx_state;
    logic [$clog2(C_WORDS)-1:0]       tx_idx;
    logic                             axis_tx_done;

    assign m_axis_tdata  = c_mem[tx_idx];
    assign m_axis_tvalid = (tx_state == TX_STREAM);
    assign m_axis_tlast  = (tx_state == TX_STREAM) && (tx_idx == C_WORDS-1);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tx_state     <= TX_IDLE;
            tx_idx       <= '0;
            axis_tx_done <= 1'b0;
        end else begin
            case (tx_state)
                TX_IDLE: begin
                    tx_idx <= '0;
                    if (tx_arm_req && done_r) begin
                        axis_tx_done <= 1'b0;
                        tx_state     <= TX_STREAM;
                    end
                end
                TX_STREAM: begin
                    if (m_axis_tvalid && m_axis_tready) begin
                        if (tx_idx == C_WORDS-1) begin
                            axis_tx_done <= 1'b1;
                            tx_state     <= TX_IDLE;
                        end else
                            tx_idx <= tx_idx + 1'b1;
                    end
                end
                default: tx_state <= TX_IDLE;
            endcase
        end
    end

    // -------------------------------------------------------------------------
    // AXI4-Lite write
    // -------------------------------------------------------------------------
    logic wr_fire;

    assign wr_fire       = s_axi_awvalid && s_axi_wvalid && !s_axi_bvalid;
    assign s_axi_awready = wr_fire;
    assign s_axi_wready  = wr_fire;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axi_bvalid <= 1'b0;
            s_axi_bresp  <= 2'b00;
            wr_en        <= 1'b0;
            wr_addr      <= '0;
            wr_data      <= '0;
            wr_strb      <= '0;
            start_req    <= 1'b0;
            clear_req    <= 1'b0;
            tx_arm_req   <= 1'b0;
        end else begin
            wr_en      <= 1'b0;
            start_req  <= 1'b0;
            clear_req  <= 1'b0;
            tx_arm_req <= 1'b0;

            if (wr_fire) begin
                wr_en        <= 1'b1;
                wr_addr      <= s_axi_awaddr[11:0];
                wr_data      <= s_axi_wdata;
                wr_strb      <= s_axi_wstrb;
                s_axi_bvalid <= 1'b1;
                s_axi_bresp  <= 2'b00;
            end else if (s_axi_bvalid && s_axi_bready) begin
                s_axi_bvalid <= 1'b0;
            end

            if (wr_en) begin
                if (wr_addr == 12'h00C) begin
                    if (wr_strb[0]) begin
                        start_req  <= wr_data[0];
                        clear_req  <= wr_data[1];
                        tx_arm_req <= wr_data[2];
                    end
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // AXI4-Lite read
    // -------------------------------------------------------------------------
    integer rbase;

    assign s_axi_arready = !s_axi_rvalid;
    assign s_axi_rresp   = 2'b00;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axi_rvalid <= 1'b0;
            s_axi_rdata  <= 32'h0;
        end else begin
            if (s_axi_arvalid && s_axi_arready) begin
                s_axi_rvalid <= 1'b1;
                if (s_axi_araddr[11:0] == 12'h000)
                    s_axi_rdata <= ID_VAL;
                else if (s_axi_araddr[11:0] == 12'h004)
                    s_axi_rdata <= VERSION_VAL;
                else if (s_axi_araddr[11:0] == 12'h008)
                    s_axi_rdata <= {28'h0, axis_tx_done, axis_rx_done, done_r, busy_r};
                else if (s_axi_araddr[11:0] == 12'h010)
                    s_axi_rdata <= N;
                else if ((s_axi_araddr[11:0] >= 12'h100) && (s_axi_araddr[11:0] < 12'h140)) begin
                    rbase = (s_axi_araddr[11:0] - 12'h100) >> 2;
                    s_axi_rdata <= {a_mem[rbase*4+3], a_mem[rbase*4+2], a_mem[rbase*4+1], a_mem[rbase*4+0]};
                end else if ((s_axi_araddr[11:0] >= 12'h200) && (s_axi_araddr[11:0] < 12'h240)) begin
                    rbase = (s_axi_araddr[11:0] - 12'h200) >> 2;
                    s_axi_rdata <= {b_mem[rbase*4+3], b_mem[rbase*4+2], b_mem[rbase*4+1], b_mem[rbase*4+0]};
                end else if ((s_axi_araddr[11:0] >= 12'h400) && (s_axi_araddr[11:0] < 12'h500)) begin
                    rbase = (s_axi_araddr[11:0] - 12'h400) >> 2;
                    s_axi_rdata <= c_mem[rbase];
                end else
                    s_axi_rdata <= 32'h0;
            end else if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
            end
        end
    end

endmodule
