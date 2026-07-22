// Single Processing Element for an output-stationary systolic array.
// On enable: acc += a_in * b_in (int8 x int8 -> int32), and forward a/b to neighbors.

module pe (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               clear,   // synchronous clear of accumulator
    input  logic               enable,  // perform MAC + forward this cycle
    input  logic signed [7:0]  a_in,
    input  logic signed [7:0]  b_in,
    output logic signed [7:0]  a_out,
    output logic signed [7:0]  b_out,
    output logic signed [31:0] acc
);

    logic signed [15:0] product;

    assign product = a_in * b_in;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc   <= '0;
            a_out <= '0;
            b_out <= '0;
        end else if (clear) begin
            acc   <= '0;
            a_out <= a_in;
            b_out <= b_in;
        end else if (enable) begin
            acc   <= acc + product;
            a_out <= a_in;
            b_out <= b_in;
        end
    end

endmodule
