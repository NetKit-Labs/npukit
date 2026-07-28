// AXI4-Lite control + BRAM A/B/C + AXIS DMA ports + skewed feed sequencer.
// Register map (byte offsets):
//   0x000 ID        R  0x4E50554B ("NPUK")
//   0x004 VERSION   R  0x00000301  (v3.1: weight-stationary + A ping-pong)
//   0x008 STATUS    R  [0] busy (GEMM|glue)
//                      [1] gemm_done  [2] axis_rx_done  [3] axis_tx_done
//                      [4] glue_done  [5] shadow_a_ready
//   0x00C CTRL      W  [0] start  [1] clear  [2] axis_tx_arm (GEMM pulses)
//   0x010 N_PARAM   R  N
//   0x014 FEATURES  R  [0] GEMM  [1] GLUE  [2] WS (A/B-only load)  [3] PP (dual-A)
//   0x018 GLUE_CTRL W  [0] start pulse  [7:4] opcode (see npukit_glue.sv)
//   0x01C GLUE_LEN  RW vector length 1..16
//   0x020 GLUE_PARAM RW RMSNorm eps (Q12) etc.
//   0x024 GLUE_COUNT R  increments each completed glue op (host sync)
//   0x028 LOAD_CFG  RW [1:0] 0=A|B(32w)  1=A-only(16w)  2=B-only(16w)
//                      writing LOAD_CFG resets AXIS RX index
//   0x100..0x13F    A  R/W  (active write bank; also via S_AXIS)
//   0x200..0x23F    B  R/W
//   0x400..0x4FF    C  R    (also readable via M_AXIS after CTRL.axis_tx_arm)
//   0x500..0x5FF    GLUE_X     R/W int32 Q12
//   0x600..0x6FF    GLUE_Y     R/W int32 Q12
//   0x700..0x7FF    GLUE_OUT   R/W int32 (Q12, or Q16 after softmax)
//   0x800..0x8FF    GLUE_GAMMA R/W int32 Q12
//
// Weight-stationary: hold B across kicks; stream A-only (LOAD_CFG=1).
// Ping-pong: dual A banks; AXIS may fill the shadow bank while GEMM runs.

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

    // AXI-Stream slave (DMA MM2S → tile payload per LOAD_CFG)
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
    localparam logic [31:0] ID_VAL       = 32'h4E50554B;
    localparam logic [31:0] VERSION_VAL  = 32'h00000301;
    // bit0=GEMM bit1=GLUE bit2=WS bit3=PP
    localparam logic [31:0] FEATURES_VAL = 32'h0000000F;
    localparam int          GLUE_LEN_MAX = 16;
    localparam logic [1:0]  LOAD_AB = 2'd0;
    localparam logic [1:0]  LOAD_A  = 2'd1;
    localparam logic [1:0]  LOAD_B  = 2'd2;

    (* ram_style = "block" *) logic signed [7:0]  a_mem [0:1][0:NN-1];
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

    logic [1:0] load_mode_r;
    logic       a_rd_bank;
    logic       a_wr_bank;
    logic       shadow_a_ready;
    logic       load_cfg_we;

    // Transformer glue (npukit_glue.sv)
    logic                       glue_start;
    logic [3:0]                 glue_opcode;
    logic [7:0]                 glue_len_r;
    logic signed [31:0]         glue_param_r;
    logic                       glue_busy;
    logic                       glue_done;
    logic [31:0]                glue_complete_count;
    logic                       glue_host_we;
    logic [1:0]                 glue_wr_bank;
    logic [$clog2(GLUE_LEN_MAX)-1:0] glue_wr_idx;
    logic signed [31:0]         glue_host_wdata;
    logic signed [31:0]         glue_host_rdata;

    logic [1:0]                      glue_host_bank;
    logic [$clog2(GLUE_LEN_MAX)-1:0] glue_host_idx;
    logic [1:0]                      glue_ar_bank;
    always_comb begin
        if (s_axi_araddr[11:0] < 12'h600)      glue_ar_bank = 2'd0;
        else if (s_axi_araddr[11:0] < 12'h700) glue_ar_bank = 2'd1;
        else if (s_axi_araddr[11:0] < 12'h800) glue_ar_bank = 2'd2;
        else                                    glue_ar_bank = 2'd3;
        if (glue_host_we) begin
            glue_host_bank = glue_wr_bank;
            glue_host_idx  = glue_wr_idx;
        end else begin
            glue_host_bank = glue_ar_bank;
            glue_host_idx  = s_axi_araddr[7:2];
        end
    end

    npukit_glue #(
        .MAX_LEN(GLUE_LEN_MAX)
    ) u_glue (
        .clk        (clk),
        .rst_n      (rst_n),
        .start      (glue_start),
        .opcode     (glue_opcode),
        .len        (glue_len_r),
        .param      (glue_param_r),
        .busy           (glue_busy),
        .done           (glue_done),
        .complete_count (glue_complete_count),
        .host_we        (glue_host_we),
        .host_bank  (glue_host_bank),
        .host_idx   (glue_host_idx),
        .host_wdata (glue_host_wdata),
        .host_rdata (glue_host_rdata)
    );

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

    assign busy = busy_r | glue_busy;
    assign done = done_r;

    // Bank swap / wr-bank prep on START (handled in mem process with shadow_a_ready)
    logic start_bank_go;
    assign start_bank_go = (state == ST_IDLE) && start_req;

    // -------------------------------------------------------------------------
    // Compute sequencer
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
                            a_west[i]  <= a_mem[a_rd_bank][i*N + (t - i)];
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
    // AXIS RX: AB / A-only / B-only; A fills may land in shadow bank while busy
    // -------------------------------------------------------------------------
    logic [$clog2(A_WORDS+B_WORDS)-1:0] rx_idx;
    logic                               axis_rx_done;
    logic [$clog2(A_WORDS+B_WORDS):0]   rx_target;

    always_comb begin
        unique case (load_mode_r)
            LOAD_A:  rx_target = A_WORDS;
            LOAD_B:  rx_target = B_WORDS;
            default: rx_target = A_WORDS + B_WORDS;
        endcase
    end

    // Idle fill always OK; during GEMM only A-only into the non-active bank.
    wire rx_idle_ok = !busy_r && !glue_busy && (state == ST_IDLE);
    wire rx_shadow_ok = (load_mode_r == LOAD_A) && busy_r && !glue_busy &&
                        (a_wr_bank != a_rd_bank) && !shadow_a_ready;
    assign s_axis_tready = rst_n && (rx_idx < rx_target) && (rx_idle_ok || rx_shadow_ok);
    // Idle A fills the active read bank; busy A-only prefetch uses the write/shadow bank.
    wire a_fill_bank = rx_shadow_ok ? a_wr_bank : a_rd_bank;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rx_idx         <= '0;
            axis_rx_done   <= 1'b0;
            load_mode_r    <= LOAD_AB;
            a_rd_bank      <= 1'b0;
            a_wr_bank      <= 1'b0;
            shadow_a_ready <= 1'b0;
            for (wi = 0; wi < NN; wi++) begin
                a_mem[0][wi] <= '0;
                a_mem[1][wi] <= '0;
                b_mem[wi]    <= '0;
                c_mem[wi]    <= '0;
            end
        end else begin
            if (clear_req)
                axis_rx_done <= 1'b0;

            // START: consume shadow A if present; always aim next fill at the other bank.
            if (start_bank_go) begin
                if (shadow_a_ready) begin
                    a_rd_bank      <= a_wr_bank;
                    a_wr_bank      <= ~a_wr_bank;
                    shadow_a_ready <= 1'b0;
                end else begin
                    a_wr_bank <= ~a_rd_bank;
                end
            end

            if (load_cfg_we) begin
                load_mode_r  <= wr_data[1:0];
                rx_idx       <= '0;
                axis_rx_done <= 1'b0;
            end

            if (capture_c) begin
                for (wi = 0; wi < NN; wi++)
                    c_mem[wi] <= c_out[wi];
            end

            if (s_axis_tvalid && s_axis_tready) begin
                if (load_mode_r == LOAD_B) begin
                    b_mem[rx_idx*4 + 0] <= s_axis_tdata[7:0];
                    b_mem[rx_idx*4 + 1] <= s_axis_tdata[15:8];
                    b_mem[rx_idx*4 + 2] <= s_axis_tdata[23:16];
                    b_mem[rx_idx*4 + 3] <= s_axis_tdata[31:24];
                end else if (load_mode_r == LOAD_A) begin
                    a_mem[a_fill_bank][rx_idx*4 + 0] <= s_axis_tdata[7:0];
                    a_mem[a_fill_bank][rx_idx*4 + 1] <= s_axis_tdata[15:8];
                    a_mem[a_fill_bank][rx_idx*4 + 2] <= s_axis_tdata[23:16];
                    a_mem[a_fill_bank][rx_idx*4 + 3] <= s_axis_tdata[31:24];
                end else if (rx_idx < A_WORDS) begin
                    a_mem[a_fill_bank][rx_idx*4 + 0] <= s_axis_tdata[7:0];
                    a_mem[a_fill_bank][rx_idx*4 + 1] <= s_axis_tdata[15:8];
                    a_mem[a_fill_bank][rx_idx*4 + 2] <= s_axis_tdata[23:16];
                    a_mem[a_fill_bank][rx_idx*4 + 3] <= s_axis_tdata[31:24];
                end else begin
                    b_mem[(rx_idx-A_WORDS)*4 + 0] <= s_axis_tdata[7:0];
                    b_mem[(rx_idx-A_WORDS)*4 + 1] <= s_axis_tdata[15:8];
                    b_mem[(rx_idx-A_WORDS)*4 + 2] <= s_axis_tdata[23:16];
                    b_mem[(rx_idx-A_WORDS)*4 + 3] <= s_axis_tdata[31:24];
                end

                if (rx_idx == rx_target - 1) begin
                    rx_idx       <= '0;
                    axis_rx_done <= 1'b1;
                    if (rx_shadow_ok)
                        shadow_a_ready <= 1'b1;
                end else
                    rx_idx <= rx_idx + 1'b1;
            end else if (wr_en && !busy_r && !load_cfg_we) begin
                if ((wr_addr >= 12'h100) && (wr_addr < 12'h140)) begin
                    base = (wr_addr - 12'h100) >> 2;
                    // MMIO always updates the active read bank (idle path).
                    if (wr_strb[0]) a_mem[a_rd_bank][base*4 + 0] <= wr_data[7:0];
                    if (wr_strb[1]) a_mem[a_rd_bank][base*4 + 1] <= wr_data[15:8];
                    if (wr_strb[2]) a_mem[a_rd_bank][base*4 + 2] <= wr_data[23:16];
                    if (wr_strb[3]) a_mem[a_rd_bank][base*4 + 3] <= wr_data[31:24];
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
            start_req      <= 1'b0;
            clear_req      <= 1'b0;
            tx_arm_req     <= 1'b0;
            load_cfg_we    <= 1'b0;
            glue_start      <= 1'b0;
            glue_opcode     <= '0;
            glue_len_r      <= 8'd16;
            glue_param_r    <= 32'sd1;
            glue_host_we    <= 1'b0;
            glue_wr_bank    <= '0;
            glue_wr_idx     <= '0;
            glue_host_wdata <= '0;
        end else begin
            wr_en        <= 1'b0;
            start_req    <= 1'b0;
            clear_req    <= 1'b0;
            tx_arm_req   <= 1'b0;
            load_cfg_we  <= 1'b0;
            glue_start   <= 1'b0;
            glue_host_we <= 1'b0;

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
                    if (wr_strb[0] && !glue_busy) begin
                        start_req  <= wr_data[0];
                        clear_req  <= wr_data[1];
                        tx_arm_req <= wr_data[2];
                    end
                end else if (wr_addr == 12'h028) begin
                    if (wr_strb[0]) load_cfg_we <= 1'b1;
                end else if (wr_addr == 12'h018) begin
                    if (wr_strb[0] && !busy_r && !glue_busy) begin
                        glue_opcode <= wr_data[7:4];
                        glue_start  <= wr_data[0];
                    end
                end else if (wr_addr == 12'h01C) begin
                    if (wr_strb[0]) glue_len_r <= wr_data[7:0];
                end else if (wr_addr == 12'h020) begin
                    glue_param_r <= wr_data;
                end else if (!glue_busy && (wr_addr >= 12'h500) && (wr_addr < 12'h900)) begin
                    glue_host_we    <= 1'b1;
                    glue_host_wdata <= wr_data;
                    glue_wr_idx     <= wr_addr[7:2];
                    if (wr_addr < 12'h600)      glue_wr_bank <= 2'd0;
                    else if (wr_addr < 12'h700) glue_wr_bank <= 2'd1;
                    else if (wr_addr < 12'h800) glue_wr_bank <= 2'd2;
                    else                        glue_wr_bank <= 2'd3;
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
                    s_axi_rdata <= {26'h0, shadow_a_ready, glue_done, axis_tx_done,
                                    axis_rx_done, done_r, (busy_r | glue_busy)};
                else if (s_axi_araddr[11:0] == 12'h010)
                    s_axi_rdata <= N;
                else if (s_axi_araddr[11:0] == 12'h014)
                    s_axi_rdata <= FEATURES_VAL;
                else if (s_axi_araddr[11:0] == 12'h01C)
                    s_axi_rdata <= {24'h0, glue_len_r};
                else if (s_axi_araddr[11:0] == 12'h020)
                    s_axi_rdata <= glue_param_r;
                else if (s_axi_araddr[11:0] == 12'h024)
                    s_axi_rdata <= glue_complete_count;
                else if (s_axi_araddr[11:0] == 12'h028)
                    s_axi_rdata <= {30'h0, load_mode_r};
                else if ((s_axi_araddr[11:0] >= 12'h100) && (s_axi_araddr[11:0] < 12'h140)) begin
                    rbase = (s_axi_araddr[11:0] - 12'h100) >> 2;
                    s_axi_rdata <= {a_mem[a_rd_bank][rbase*4+3], a_mem[a_rd_bank][rbase*4+2],
                                    a_mem[a_rd_bank][rbase*4+1], a_mem[a_rd_bank][rbase*4+0]};
                end else if ((s_axi_araddr[11:0] >= 12'h200) && (s_axi_araddr[11:0] < 12'h240)) begin
                    rbase = (s_axi_araddr[11:0] - 12'h200) >> 2;
                    s_axi_rdata <= {b_mem[rbase*4+3], b_mem[rbase*4+2], b_mem[rbase*4+1],
                                    b_mem[rbase*4+0]};
                end else if ((s_axi_araddr[11:0] >= 12'h400) && (s_axi_araddr[11:0] < 12'h500)) begin
                    rbase = (s_axi_araddr[11:0] - 12'h400) >> 2;
                    s_axi_rdata <= c_mem[rbase];
                end else if ((s_axi_araddr[11:0] >= 12'h500) && (s_axi_araddr[11:0] < 12'h900))
                    s_axi_rdata <= glue_host_rdata;
                else
                    s_axi_rdata <= 32'h0;
            end else if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
            end
        end
    end

endmodule
