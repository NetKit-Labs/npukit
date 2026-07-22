// Verilog wrapper so Vivado Block Design can module-reference NpuKit.
// (BD module references cannot use a SystemVerilog file as the reference top.)

module npukit_pl (
    input  wire       clk,
    input  wire       btn0,
    output wire [3:0] led
);

    npukit_top u_npukit (
        .clk (clk),
        .btn0(btn0),
        .led (led)
    );

endmodule
