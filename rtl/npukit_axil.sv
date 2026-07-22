// AXI4-Lite control + A/B/C memories + skewed feed sequencer for systolic_array.
// Register map (byte offsets, 32-bit words):
//   0x000 ID       R  0x4E50554B ("NPUK")
//   0x004 VERSION  R  0x00000100
//   0x008 STATUS   R  [0] busy  [1] done
//   0x00C CTRL     W  [0] start  [1] clear   (write-1-to-pulse)
//   0x010 N_PARAM  R  N
//   0x100..0x13F   A  R/W  16 x uint32, packed int8 row-major (4 bytes/word)
//   0x200..0x23F   B  R/W  same packing
//   0x400..0x4FF   C  R    64 x int32 row-major

module npukit_axil #(
    parameter int N          = 8,
    parameter int ADDR_WIDTH = 16
) (
    input  logic                  clk,
    input  logic                  rst_n,

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

    output logic                  busy,
    output logic                  done
);

    localparam int NN       = N * N;
    localparam int FEED_CYC = 3 * N - 2;
    localparam logic [31:0] ID_VAL      = 32'h4E50554B;
    localparam logic [31:0] VERSION_VAL = 32'h00000100;

    logic signed [7:0]  a_mem [0:NN-1];
    logic signed [7:0]  b_mem [0:NN-1];
    logic signed [31:0] c_mem [0:NN-1];

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

    typedef enum logic [2:0] {
        ST_IDLE  = 3'd0,
        ST_CLEAR = 3'd1,
        ST_RUN   = 3'd2,
        ST_WAIT  = 3'd3,  // let last enable MAC commit before latching C
        ST_DONE  = 3'd4
    } state_t;

    state_t                        state;
    logic [$clog2(FEED_CYC)-1:0]   t;
    logic                          do_run_after_clear;
    logic                          busy_r;
    logic                          done_r;
    integer                        i;

    assign busy = busy_r;
    assign done = done_r;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state              <= ST_IDLE;
            t                  <= '0;
            clear              <= 1'b0;
            enable             <= 1'b0;
            busy_r             <= 1'b0;
            done_r             <= 1'b0;
            do_run_after_clear <= 1'b0;
            for (i = 0; i < N; i++) begin
                a_west[i]  <= '0;
                b_north[i] <= '0;
            end
            for (i = 0; i < NN; i++)
                c_mem[i] <= '0;
        end else begin
            clear  <= 1'b0;
            enable <= 1'b0;
            for (i = 0; i < N; i++) begin
                a_west[i]  <= '0;
                b_north[i] <= '0;
            end

            case (state)
                ST_IDLE: begin
                    busy_r <= 1'b0;
                    if (clear_req || start_req) begin
                        busy_r             <= 1'b1;
                        done_r             <= 1'b0;
                        do_run_after_clear <= start_req;
                        state              <= ST_CLEAR;
                    end
                end

                ST_CLEAR: begin
                    busy_r <= 1'b1;
                    clear  <= 1'b1;
                    t      <= '0;
                    if (do_run_after_clear)
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
                    // Previous cycle scheduled enable=1 for t=last; that MAC commits now.
                    busy_r <= 1'b1;
                    state  <= ST_DONE;
                end

                ST_DONE: begin
                    for (i = 0; i < NN; i++)
                        c_mem[i] <= c_out[i];
                    done_r <= 1'b1;
                    busy_r <= 1'b0;
                    state  <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

    // -------------------------------------------------------------------------
    // AXI4-Lite write
    // -------------------------------------------------------------------------
    logic        wr_fire;
    logic        wr_en;
    logic [11:0] wr_addr;
    logic [31:0] wr_data;
    logic [3:0]  wr_strb;
    integer      wi;
    integer      base;

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
            for (wi = 0; wi < NN; wi++) begin
                a_mem[wi] <= '0;
                b_mem[wi] <= '0;
            end
        end else begin
            wr_en     <= 1'b0;
            start_req <= 1'b0;
            clear_req <= 1'b0;

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
                        start_req <= wr_data[0];
                        clear_req <= wr_data[1];
                    end
                end else if ((wr_addr >= 12'h100) && (wr_addr < 12'h140) && !busy_r) begin
                    base = (wr_addr - 12'h100) >> 2;
                    if (wr_strb[0]) a_mem[base*4 + 0] <= wr_data[7:0];
                    if (wr_strb[1]) a_mem[base*4 + 1] <= wr_data[15:8];
                    if (wr_strb[2]) a_mem[base*4 + 2] <= wr_data[23:16];
                    if (wr_strb[3]) a_mem[base*4 + 3] <= wr_data[31:24];
                end else if ((wr_addr >= 12'h200) && (wr_addr < 12'h240) && !busy_r) begin
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
    // AXI4-Lite read
    // -------------------------------------------------------------------------
    logic [11:0] ar_addr_q;
    integer      rbase;

    assign s_axi_arready = !s_axi_rvalid;
    assign s_axi_rresp   = 2'b00;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axi_rvalid <= 1'b0;
            s_axi_rdata  <= 32'h0;
            ar_addr_q    <= 12'h0;
        end else begin
            if (s_axi_arvalid && s_axi_arready) begin
                ar_addr_q    <= s_axi_araddr[11:0];
                s_axi_rvalid <= 1'b1;

                if (s_axi_araddr[11:0] == 12'h000)
                    s_axi_rdata <= ID_VAL;
                else if (s_axi_araddr[11:0] == 12'h004)
                    s_axi_rdata <= VERSION_VAL;
                else if (s_axi_araddr[11:0] == 12'h008)
                    s_axi_rdata <= {30'h0, done_r, busy_r};
                else if (s_axi_araddr[11:0] == 12'h010)
                    s_axi_rdata <= N;
                else if ((s_axi_araddr[11:0] >= 12'h100) && (s_axi_araddr[11:0] < 12'h140)) begin
                    rbase = (s_axi_araddr[11:0] - 12'h100) >> 2;
                    s_axi_rdata <= {
                        a_mem[rbase*4 + 3],
                        a_mem[rbase*4 + 2],
                        a_mem[rbase*4 + 1],
                        a_mem[rbase*4 + 0]
                    };
                end else if ((s_axi_araddr[11:0] >= 12'h200) && (s_axi_araddr[11:0] < 12'h240)) begin
                    rbase = (s_axi_araddr[11:0] - 12'h200) >> 2;
                    s_axi_rdata <= {
                        b_mem[rbase*4 + 3],
                        b_mem[rbase*4 + 2],
                        b_mem[rbase*4 + 1],
                        b_mem[rbase*4 + 0]
                    };
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
