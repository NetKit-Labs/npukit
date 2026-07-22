`timescale 1ns / 1ps

module pe_tb;

    logic               clk;
    logic               rst_n;
    logic               clear;
    logic               enable;
    logic signed [7:0]  a_in;
    logic signed [7:0]  b_in;
    logic signed [7:0]  a_out;
    logic signed [7:0]  b_out;
    logic signed [31:0] acc;

    int errors;

    pe dut (
        .clk   (clk),
        .rst_n (rst_n),
        .clear (clear),
        .enable(enable),
        .a_in  (a_in),
        .b_in  (b_in),
        .a_out (a_out),
        .b_out (b_out),
        .acc   (acc)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    task automatic check(string name, logic signed [31:0] got, logic signed [31:0] exp);
        if (got !== exp) begin
            $display("FAIL %s: got %0d exp %0d", name, got, exp);
            errors++;
        end else begin
            $display("PASS %s: %0d", name, got);
        end
    endtask

    initial begin
        errors = 0;
        rst_n  = 1'b0;
        clear  = 1'b0;
        enable = 1'b0;
        a_in   = '0;
        b_in   = '0;

        repeat (3) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        // clear
        clear = 1'b1;
        @(posedge clk);
        clear = 1'b0;
        @(posedge clk);
        check("after clear", acc, 0);

        // 3 * 4 = 12
        a_in   = 8'sd3;
        b_in   = 8'sd4;
        enable = 1'b1;
        @(posedge clk);
        enable = 1'b0;
        @(posedge clk);
        check("3*4", acc, 32'sd12);
        check("a forward", a_out, 8'sd3);
        check("b forward", b_out, 8'sd4);

        // accumulate: + (-2)*5 = -10 → 2
        a_in   = -8'sd2;
        b_in   = 8'sd5;
        enable = 1'b1;
        @(posedge clk);
        enable = 1'b0;
        @(posedge clk);
        check("acc after -2*5", acc, 32'sd2);

        // clear again
        clear = 1'b1;
        @(posedge clk);
        clear = 1'b0;
        @(posedge clk);
        check("clear again", acc, 0);

        if (errors == 0)
            $display("pe_tb: ALL PASS");
        else
            $display("pe_tb: %0d FAIL(s)", errors);

        $finish;
    end

endmodule
