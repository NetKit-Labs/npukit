// 8x8 output-stationary int8 systolic array.
// A flows left -> right; B flows top -> bottom; C accumulates in each PE.
// c_out is row-major flat: index = row*N + col.

module systolic_array #(
    parameter int N = 8
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    clear,
    input  logic                    enable,
    input  logic signed [7:0]       a_west  [0:N-1],
    input  logic signed [7:0]       b_north [0:N-1],
    output wire  signed [31:0]      c_out   [0:N*N-1]
);

    wire signed [7:0]  a_link [0:N*(N+1)-1];
    wire signed [7:0]  b_link [0:(N+1)*N-1];
    wire signed [31:0] acc_w  [0:N*N-1];

    genvar r, c, idx;
    generate
        for (r = 0; r < N; r = r + 1) begin : g_aw
            assign a_link[r*(N+1) + 0] = a_west[r];
        end
        for (c = 0; c < N; c = c + 1) begin : g_bn
            assign b_link[0*N + c] = b_north[c];
        end

        for (r = 0; r < N; r = r + 1) begin : g_r
            for (c = 0; c < N; c = c + 1) begin : g_c
                pe u_pe (
                    .clk   (clk),
                    .rst_n (rst_n),
                    .clear (clear),
                    .enable(enable),
                    .a_in  (a_link[r*(N+1) + c]),
                    .b_in  (b_link[r*N + c]),
                    .a_out (a_link[r*(N+1) + (c+1)]),
                    .b_out (b_link[(r+1)*N + c]),
                    .acc   (acc_w[r*N + c])
                );
            end
        end

        for (idx = 0; idx < N*N; idx = idx + 1) begin : g_cout
            assign c_out[idx] = acc_w[idx];
        end
    endgenerate

endmodule
