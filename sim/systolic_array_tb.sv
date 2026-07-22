`timescale 1ns / 1ps

// Feed an 8x8 output-stationary array with skewed A (west) and B (north)
// streams and check C = A * B for a simple known pair of matrices.

module systolic_array_tb;

    localparam int N = 8;

    logic               clk;
    logic               rst_n;
    logic               clear;
    logic               enable;
    logic signed [7:0]  a_west  [0:N-1];
    logic signed [7:0]  b_north [0:N-1];
    wire  signed [31:0] c_out   [0:N*N-1];

    logic signed [7:0]  A [0:N-1][0:N-1];
    logic signed [7:0]  B [0:N-1][0:N-1];
    logic signed [31:0] C_ref [0:N-1][0:N-1];

    int errors;
    int t;
    int i, j, k;

    systolic_array #(
        .N(N)
    ) dut (
        .clk    (clk),
        .rst_n  (rst_n),
        .clear  (clear),
        .enable (enable),
        .a_west (a_west),
        .b_north(b_north),
        .c_out  (c_out)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    initial begin
        errors = 0;
        rst_n  = 1'b0;
        clear  = 1'b0;
        enable = 1'b0;

        for (i = 0; i < N; i++) begin
            a_west[i]  = '0;
            b_north[i] = '0;
        end

        // A[i][k] = i+1, B[k][j] = 1 → C[i][j] = N*(i+1)
        for (i = 0; i < N; i++) begin
            for (j = 0; j < N; j++) begin
                A[i][j] = 8'(i + 1);
                B[i][j] = 8'sd1;
            end
        end

        for (i = 0; i < N; i++) begin
            for (j = 0; j < N; j++) begin
                C_ref[i][j] = 0;
                for (k = 0; k < N; k++)
                    C_ref[i][j] += A[i][k] * B[k][j];
            end
        end

        repeat (3) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        clear = 1'b1;
        @(posedge clk);
        clear = 1'b0;
        @(posedge clk);

        // Skewed feed long enough for PE[N-1][N-1] to see all K terms (3N-2 cycles)
        for (t = 0; t < 3*N-2; t++) begin
            for (i = 0; i < N; i++) begin
                if ((t >= i) && ((t - i) < N))
                    a_west[i] = A[i][t-i];
                else
                    a_west[i] = '0;

                if ((t >= i) && ((t - i) < N))
                    b_north[i] = B[t-i][i];
                else
                    b_north[i] = '0;
            end
            enable = 1'b1;
            @(posedge clk);
        end

        enable = 1'b0;
        for (i = 0; i < N; i++) begin
            a_west[i]  = '0;
            b_north[i] = '0;
        end

        repeat (2) @(posedge clk);

        for (i = 0; i < N; i++) begin
            for (j = 0; j < N; j++) begin
                if (c_out[i*N + j] !== C_ref[i][j]) begin
                    $display("FAIL C[%0d][%0d]: got %0d exp %0d",
                             i, j, c_out[i*N + j], C_ref[i][j]);
                    errors++;
                end
            end
        end

        if (errors == 0)
            $display("systolic_array_tb: ALL PASS (8x8 int8 matmul)");
        else
            $display("systolic_array_tb: %0d FAIL(s)", errors);

        $finish;
    end

endmodule
