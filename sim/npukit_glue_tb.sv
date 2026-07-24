`timescale 1ns / 1ps

// Unit check for transformer glue: residual + softmax smoke.
module npukit_glue_tb;

    localparam int MAX_LEN = 16;

    logic                       clk;
    logic                       rst_n;
    logic                       start;
    logic [3:0]                 opcode;
    logic [7:0]                 len;
    logic signed [31:0]         param;
    logic                       busy;
    logic                       done;
    logic [31:0]                complete_count;
    logic                       host_we;
    logic [1:0]                 host_bank;
    logic [$clog2(MAX_LEN)-1:0] host_idx;
    logic signed [31:0]         host_wdata;
    logic signed [31:0]         host_rdata;

    int errors;
    int i;
    logic signed [31:0] got;

    npukit_glue #(.MAX_LEN(MAX_LEN)) dut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .opcode(opcode), .len(len), .param(param),
        .busy(busy), .done(done), .complete_count(complete_count),
        .host_we(host_we), .host_bank(host_bank), .host_idx(host_idx),
        .host_wdata(host_wdata), .host_rdata(host_rdata)
    );

    always #5 clk = ~clk;

    task automatic write_bank(input [1:0] bank, input int idx, input logic signed [31:0] val);
        begin
            @(posedge clk);
            host_we    <= 1'b1;
            host_bank  <= bank;
            host_idx   <= idx[$clog2(MAX_LEN)-1:0];
            host_wdata <= val;
            @(posedge clk);
            host_we <= 1'b0;
        end
    endtask

    task automatic read_out(input int idx, output logic signed [31:0] val);
        begin
            @(posedge clk);
            host_bank <= 2'd2;
            host_idx  <= idx[$clog2(MAX_LEN)-1:0];
            #1 val = host_rdata;
        end
    endtask

    task automatic run_op(input [3:0] op, input int n);
        begin
            @(posedge clk);
            opcode <= op;
            len    <= n[7:0];
            start  <= 1'b1;
            @(posedge clk);
            start <= 1'b0;
            wait (done);
            @(posedge clk);
        end
    endtask

    initial begin
        clk = 0;
        rst_n = 0;
        start = 0;
        opcode = 0;
        len = 8;
        param = 1;
        host_we = 0;
        host_bank = 0;
        host_idx = 0;
        host_wdata = 0;
        errors = 0;

        repeat (4) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);

        // Residual: X=[1,2,3,4], Y=[10,20,30,40] in Q12
        for (i = 0; i < 4; i++) begin
            write_bank(2'd0, i, (i + 1) <<< 12);
            write_bank(2'd1, i, ((i + 1) * 10) <<< 12);
        end
        run_op(4'h1, 4);
        for (i = 0; i < 4; i++) begin
            read_out(i, got);
            if (got !== (((i + 1) + (i + 1) * 10) <<< 12)) begin
                $display("RESIDUAL FAIL i=%0d got=%0d", i, got);
                errors++;
            end
        end

        // Softmax smoke: large first logit → ~one-hot on index 0
        write_bank(2'd0, 0, 32'sd8192);  // 2.0
        write_bank(2'd0, 1, 32'sd0);
        write_bank(2'd0, 2, -32'sd4096);
        write_bank(2'd0, 3, -32'sd8192);
        run_op(4'h4, 4);
        read_out(0, got);
        if (got < 32'sd40000) begin
            $display("SOFTMAX FAIL: expected dominant class0, got %0d", got);
            errors++;
        end

        if (errors == 0) $display("npukit_glue_tb: ALL PASS");
        else $display("npukit_glue_tb: FAIL (%0d)", errors);
        $finish;
    end
endmodule
