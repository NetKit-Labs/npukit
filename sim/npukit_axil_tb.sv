`timescale 1ns / 1ps

// AXI4-Lite host path check: write A/B, START, poll DONE, read C vs NumPy-equivalent.

module npukit_axil_tb;

    localparam int N = 8;
    localparam int NN = N * N;

    logic        clk;
    logic        rst_n;

    logic [15:0] s_axi_awaddr;
    logic        s_axi_awvalid;
    logic        s_axi_awready;
    logic [31:0] s_axi_wdata;
    logic [3:0]  s_axi_wstrb;
    logic        s_axi_wvalid;
    logic        s_axi_wready;
    logic [1:0]  s_axi_bresp;
    logic        s_axi_bvalid;
    logic        s_axi_bready;

    logic [15:0] s_axi_araddr;
    logic        s_axi_arvalid;
    logic        s_axi_arready;
    logic [31:0] s_axi_rdata;
    logic [1:0]  s_axi_rresp;
    logic        s_axi_rvalid;
    logic        s_axi_rready;

    logic [31:0] s_axis_tdata;
    logic        s_axis_tvalid;
    logic        s_axis_tready;
    logic        s_axis_tlast;
    logic [31:0] m_axis_tdata;
    logic        m_axis_tvalid;
    logic        m_axis_tready;
    logic        m_axis_tlast;

    logic        busy;
    logic        done;

    logic signed [7:0]  A [0:NN-1];
    logic signed [7:0]  B [0:NN-1];
    logic signed [31:0] C_ref [0:NN-1];
    logic signed [31:0] C_got [0:NN-1];

    int errors;
    int i, j, k;
    int status;

    npukit_axil #(
        .N(N)
    ) dut (
        .clk          (clk),
        .rst_n        (rst_n),
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

    initial clk = 1'b0;
    always #5 clk = ~clk;

    task automatic axi_write(input int addr, input logic [31:0] data);
        @(posedge clk);
        s_axi_awaddr  <= addr;
        s_axi_awvalid <= 1'b1;
        s_axi_wdata   <= data;
        s_axi_wstrb   <= 4'hF;
        s_axi_wvalid  <= 1'b1;
        s_axi_bready  <= 1'b1;
        while (!(s_axi_awready && s_axi_wready))
            @(posedge clk);
        @(posedge clk);
        s_axi_awvalid <= 1'b0;
        s_axi_wvalid  <= 1'b0;
        while (!s_axi_bvalid)
            @(posedge clk);
        @(posedge clk);
        s_axi_bready <= 1'b0;
    endtask

    task automatic axi_read(input int addr, output logic [31:0] data);
        @(posedge clk);
        s_axi_araddr  <= addr;
        s_axi_arvalid <= 1'b1;
        s_axi_rready  <= 1'b1;
        while (!s_axi_arready)
            @(posedge clk);
        @(posedge clk);
        s_axi_arvalid <= 1'b0;
        while (!s_axi_rvalid)
            @(posedge clk);
        data = s_axi_rdata;
        @(posedge clk);
        s_axi_rready <= 1'b0;
    endtask

    initial begin
        errors        = 0;
        rst_n         = 1'b0;
        s_axi_awaddr  = '0;
        s_axi_awvalid = 1'b0;
        s_axi_wdata   = '0;
        s_axi_wstrb   = '0;
        s_axi_wvalid  = 1'b0;
        s_axi_bready  = 1'b0;
        s_axi_araddr  = '0;
        s_axi_arvalid = 1'b0;
        s_axi_rready  = 1'b0;
        s_axis_tdata  = '0;
        s_axis_tvalid = 1'b0;
        s_axis_tlast  = 1'b0;
        m_axis_tready = 1'b0;

        // Same stimulus as systolic_array_tb
        for (i = 0; i < N; i++) begin
            for (j = 0; j < N; j++) begin
                A[i*N + j] = 8'(i + 1);
                B[i*N + j] = 8'sd1;
            end
        end
        for (i = 0; i < N; i++) begin
            for (j = 0; j < N; j++) begin
                C_ref[i*N + j] = 0;
                for (k = 0; k < N; k++)
                    C_ref[i*N + j] += A[i*N + k] * B[k*N + j];
            end
        end

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        axi_read(32'h000, status);
        if (status !== 32'h4E50554B) begin
            $display("FAIL ID: got %h", status);
            errors++;
        end

        // Pack A/B as little-endian int8 x4 per word
        for (i = 0; i < NN; i += 4)
            axi_write(32'h100 + i, {A[i+3], A[i+2], A[i+1], A[i+0]});
        for (i = 0; i < NN; i += 4)
            axi_write(32'h200 + i, {B[i+3], B[i+2], B[i+1], B[i+0]});

        axi_write(32'h00C, 32'h2); // CLEAR
        axi_write(32'h00C, 32'h1); // START

        // Poll STATUS.done
        status = 0;
        i = 0;
        while ((i < 200) && !status[1]) begin
            axi_read(32'h008, status);
            i++;
        end
        if (!status[1]) begin
            $display("FAIL: timed out waiting for done (status=%h busy=%b done=%b)",
                     status, busy, done);
            errors++;
        end

        for (i = 0; i < NN; i++) begin
            axi_read(32'h400 + (i * 4), status);
            C_got[i] = status;
            if (C_got[i] !== C_ref[i]) begin
                $display("FAIL C[%0d]: got %0d exp %0d", i, C_got[i], C_ref[i]);
                errors++;
            end
        end

        // A second START must retain C: this is the K-tile accumulation path.
        axi_write(32'h00C, 32'h1);
        status = 0;
        i = 0;
        while ((i < 200) && !status[1]) begin
            axi_read(32'h008, status);
            i++;
        end
        for (i = 0; i < NN; i++) begin
            axi_read(32'h400 + (i * 4), status);
            if ($signed(status) !== (C_ref[i] * 2)) begin
                $display("FAIL accumulated C[%0d]: got %0d exp %0d",
                         i, $signed(status), C_ref[i] * 2);
                errors++;
            end
        end

        // -----------------------------------------------------------------
        // Layer-resident W: AXIS LOAD_W, A via MMIO, CTRL.use_wmem
        // -----------------------------------------------------------------
        begin : wmem_test
            logic signed [7:0]  Ww [0:NN-1];
            logic signed [7:0]  Aw [0:NN-1];
            logic signed [31:0] Cw_ref [0:NN-1];
            int wi, wj, wk;
            logic [31:0] wdata;

            for (wi = 0; wi < N; wi++) begin
                for (wj = 0; wj < N; wj++) begin
                    Aw[wi*N + wj] = 8'(wi - wj);
                    Ww[wi*N + wj] = 8'(wj + 2);
                end
            end
            for (wi = 0; wi < N; wi++)
                for (wj = 0; wj < N; wj++) begin
                    Cw_ref[wi*N + wj] = 0;
                    for (wk = 0; wk < N; wk++)
                        Cw_ref[wi*N + wj] += int'(Aw[wi*N + wk]) * int'(Ww[wk*N + wj]);
                end

            axi_read(32'h004, status);
            if (status !== 32'h00000302) begin
                $display("FAIL VERSION: got %h exp 302", status);
                errors++;
            end
            axi_read(32'h014, status);
            if (!(status[4])) begin
                $display("FAIL FEATURES.WMEM clear: %h", status);
                errors++;
            end

            axi_write(32'h02C, {16'd8, 16'd8}); // W_SHAPE N<<16|K
            axi_write(32'h030, 32'd0);           // TILE_KJ = (0,0)
            axi_write(32'h028, 32'd3);           // LOAD_W
            // LOAD_CFG / W_SHAPE take effect one cycle after BRESP (wr_en path).
            repeat (4) @(posedge clk);

            // Stream W over AXIS (16 words) — one beat per iteration (no double-fire)
            for (wi = 0; wi < NN; wi += 4) begin
                @(posedge clk);
                s_axis_tdata  <= {Ww[wi+3], Ww[wi+2], Ww[wi+1], Ww[wi+0]};
                s_axis_tvalid <= 1'b1;
                s_axis_tlast  <= (wi + 4 >= NN);
                while (!s_axis_tready) @(posedge clk);
            end
            @(posedge clk);
            s_axis_tvalid <= 1'b0;
            s_axis_tlast  <= 1'b0;
            // Wait AXIS RX done
            status = 0;
            wi = 0;
            while ((wi < 50) && !status[2]) begin
                axi_read(32'h008, status);
                wi++;
            end
            if (!status[2]) begin
                $display("FAIL wmem: AXIS RX not done");
                errors++;
            end

            for (wi = 0; wi < NN; wi += 4)
                axi_write(32'h100 + wi, {Aw[wi+3], Aw[wi+2], Aw[wi+1], Aw[wi+0]});
            repeat (2) @(posedge clk);

            axi_write(32'h00C, 32'h2); // CLEAR (drops stale DONE)
            repeat (2) @(posedge clk);
            axi_write(32'h00C, 32'h9); // START|USE_WMEM
            // Wait busy then done (avoid latching prior DONE).
            status = 0;
            wi = 0;
            while ((wi < 50) && !status[0]) begin
                axi_read(32'h008, status);
                wi++;
            end
            wi = 0;
            while ((wi < 400) && !(status[1] && !status[0])) begin
                axi_read(32'h008, status);
                wi++;
            end
            if (!(status[1] && !status[0])) begin
                $display("FAIL wmem: timed out (status=%h)", status);
                errors++;
            end
            for (wi = 0; wi < NN; wi++) begin
                axi_read(32'h400 + (wi * 4), wdata);
                if ($signed(wdata) !== Cw_ref[wi]) begin
                    $display("FAIL wmem C[%0d]: got %0d exp %0d",
                             wi, $signed(wdata), Cw_ref[wi]);
                    errors++;
                end
            end
        end

        if (errors == 0)
            $display("npukit_axil_tb: ALL PASS");
        else
            $display("npukit_axil_tb: %0d FAIL(s)", errors);

        $finish;
    end

endmodule
