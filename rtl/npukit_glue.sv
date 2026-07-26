// =============================================================================
// npukit_glue — transformer epilogue ops (separate from the GEMM array)
// =============================================================================
// Residual / GELU / RMSNorm / Softmax.  RoPE, masks, reshape stay on the A9.
//
// Fixed-point (match host/npukit_transformer.py):
//   X,Y,OUT,GAMMA : int32 Q12   Softmax OUT : int32 Q16
//
// Opcodes: 1=RESIDUAL 2=GELU 3=RMSNORM 4=SOFTMAX
// LUT address / divide / isqrt / dual-mul are multi-cycle for 100 MHz.
// =============================================================================

module npukit_glue #(
    parameter int MAX_LEN = 16  // keep small for 100 MHz mux timing on Z7020
) (
    input  logic                       clk,
    input  logic                       rst_n,
    input  logic                       start,
    input  logic [3:0]                 opcode,
    input  logic [7:0]                 len,
    input  logic signed [31:0]         param,
    output logic                       busy,
    output logic                       done,
    output logic [31:0]                complete_count,
    input  logic                       host_we,
    input  logic [1:0]                 host_bank,
    input  logic [$clog2(MAX_LEN)-1:0] host_idx,
    input  logic signed [31:0]         host_wdata,
    output logic signed [31:0]         host_rdata
);

    localparam int IDX_W = $clog2(MAX_LEN);
    localparam logic [3:0] OP_RESIDUAL = 4'h1;
    localparam logic [3:0] OP_GELU     = 4'h2;
    localparam logic [3:0] OP_RMSNORM  = 4'h3;
    localparam logic [3:0] OP_SOFTMAX  = 4'h4;
    localparam int Q12 = 12;
    localparam logic signed [31:0] ONE_Q12   = 32'sd4096;
    localparam logic signed [31:0] FOUR_Q12  = 32'sd16384;
    localparam logic signed [31:0] EIGHT_Q12 = 32'sd32768;

    (* ram_style = "registers" *) logic signed [31:0] x_mem     [0:MAX_LEN-1];
    (* ram_style = "registers" *) logic signed [31:0] y_mem     [0:MAX_LEN-1];
    (* ram_style = "registers" *) logic signed [31:0] out_mem   [0:MAX_LEN-1];
    (* ram_style = "registers" *) logic signed [31:0] gamma_mem [0:MAX_LEN-1];
    (* rom_style = "distributed" *) logic signed [31:0] gelu_lut [0:255];
    (* rom_style = "distributed" *) logic signed [31:0] exp_lut  [0:255];

    function automatic logic signed [31:0] gelu_poly_q12(input logic signed [31:0] x_q12);
        logic signed [31:0] xc;
        logic signed [63:0] x, x2, x3, u, t, y, absu, den;
        begin
            xc = x_q12;
            if (xc > FOUR_Q12)  xc = FOUR_Q12;
            if (xc < -FOUR_Q12) xc = -FOUR_Q12;
            x  = xc;
            x2 = (x * x) >>> Q12;
            x3 = (x2 * x) >>> Q12;
            u = (3269 * x + 146 * x3) >>> Q12;
            absu = (u[63]) ? -u : u;
            den  = ONE_Q12 + absu[31:0];
            t    = (u <<< Q12) / den;
            y    = (x * (ONE_Q12 + t)) >>> (Q12 + 1);
            return y[31:0];
        end
    endfunction

    function automatic logic signed [31:0] exp_poly_q16(input logic signed [31:0] t_q12);
        logic signed [31:0] clamped;
        logic signed [63:0] s, base, mul;
        logic [31:0] mag, sh, frac;
        begin
            clamped = t_q12;
            if (clamped > 0) clamped = 0;
            if (clamped < -EIGHT_Q12) clamped = -EIGHT_Q12;
            s    = (5909 * clamped) >>> Q12;
            mag  = (-s[31:0]);
            sh   = mag >> Q12;
            if (sh > 15) sh = 15;
            frac = mag & 32'h0000_0FFF;
            base = 64'sd65536 - ((2839 * frac) >>> (Q12 - 4));
            if (base < 0) base = 0;
            mul = base >>> sh;
            return mul[31:0];
        end
    endfunction

    // Map Q12 span of 2^15 onto 256 bins: (x + bias) >> 7.
    // Saturate at 255 — (0 + 32768) >> 7 == 256 must not wrap to 0.
    function automatic logic [7:0] gelu_index(input logic signed [31:0] x);
        logic [31:0] s;
        begin
            s = $unsigned(x + FOUR_Q12);
            return (s >= 32'd32768) ? 8'd255 : s[14:7];
        end
    endfunction

    function automatic logic [7:0] exp_index(input logic signed [31:0] t);
        logic [31:0] s;
        begin
            s = $unsigned(t + EIGHT_Q12);
            return (s >= 32'd32768) ? 8'd255 : s[14:7];
        end
    endfunction

    integer gi;
    initial begin
        for (gi = 0; gi < 256; gi = gi + 1) begin
            gelu_lut[gi] = gelu_poly_q12(-FOUR_Q12 + ((8 * ONE_Q12 * gi) >>> 8));
            exp_lut[gi]  = exp_poly_q16(-EIGHT_Q12 + ((8 * ONE_Q12 * gi) >>> 8));
        end
    end

    always_ff @(posedge clk) begin
        if (host_we && !busy) begin
            unique case (host_bank)
                2'd0: x_mem[host_idx]     <= host_wdata;
                2'd1: y_mem[host_idx]     <= host_wdata;
                2'd3: gamma_mem[host_idx] <= host_wdata;
                default: ;
            endcase
        end
    end

    always_comb begin
        unique case (host_bank)
            2'd0: host_rdata = x_mem[host_idx];
            2'd1: host_rdata = y_mem[host_idx];
            2'd2: host_rdata = out_mem[host_idx];
            2'd3: host_rdata = gamma_mem[host_idx];
            default: host_rdata = '0;
        endcase
    end

    typedef enum logic [5:0] {
        ST_IDLE,
        ST_RES,
        ST_GELU_LOAD,
        ST_GELU_ADDR,
        ST_GELU_WR,
        ST_RMS_LOAD,
        ST_RMS_SUM,
        ST_RMS_SUM2,
        ST_RMS_ACC,
        ST_RMS_MEAN,
        ST_ISQRT_MID,
        ST_ISQRT_SQ,
        ST_ISQRT_UPD,
        ST_INV_LAUNCH,
        ST_INV_WAIT,
        ST_RMS_MUL1,
        ST_RMS_MUL2,
        ST_RMS_MUL2B,
        ST_RMS_MUL3,
        ST_RMS_MUL3B,
        ST_RMS_MUL3C,
        ST_RMS_WR,
        ST_RES_LOAD,
        ST_SM_MAX_LOAD,
        ST_SM_MAX,
        ST_SM_LOAD,
        ST_SM_CLAMP,
        ST_SM_ADDR,
        ST_SM_ACC,
        ST_SM_LAUNCH,
        ST_SM_WAIT,
        ST_DONE
    } state_t;

    state_t                 state;
    logic [IDX_W:0]         len_r;
    logic [IDX_W:0]         idx;
    logic signed [31:0]     param_r;
    logic signed [63:0]     acc64;
    logic signed [31:0]     max_v;
    logic signed [31:0]     inv_rms_q12;
    logic [31:0]            sum_exp;
    logic                   busy_r, done_r;
    logic [31:0]            complete_count_r;

    // 32÷32 restoring divider as two 32-bit regs (not one fat 64-bit wire).
    logic [31:0]            div_num;
    logic [31:0]            div_denom;
    logic [31:0]            div_quot;
    logic [31:0]            div_rem;
    logic [5:0]             div_bit;
    logic                   div_busy;
    logic [32:0]            div_trial;

    logic [63:0]            mean_u;
    logic [31:0]            isqrt_lo, isqrt_hi, isqrt_mid;
    logic [5:0]             isqrt_step;
    logic [63:0]            isqrt_sq;
    logic                   isqrt_ge;

    logic signed [31:0]     x_r, y_r, g_r, o_r;
    logic signed [31:0]     prod_r;  // Q12 after first mul (keeps DSP single-cycle)
    logic signed [31:0]     mul_a, mul_b;
    logic signed [63:0]     mul64;
    logic [7:0]             lut_addr;
    logic                   gelu_passthru;
    logic                   gelu_zero;

    assign busy = busy_r;
    assign done = done_r;
    assign complete_count = complete_count_r;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= ST_IDLE;
            busy_r           <= 1'b0;
            done_r           <= 1'b0;
            complete_count_r <= '0;
            len_r            <= '0;
            idx              <= '0;
            param_r          <= 32'sd1;
            acc64            <= '0;
            max_v            <= '0;
            inv_rms_q12      <= ONE_Q12;
            sum_exp          <= '0;
            div_busy         <= 1'b0;
            div_num          <= '0;
            div_denom        <= 32'd1;
            div_quot         <= '0;
            div_rem          <= '0;
            div_bit          <= '0;
            mean_u           <= '0;
            isqrt_lo         <= '0;
            isqrt_hi         <= '0;
            isqrt_mid        <= '0;
            isqrt_sq         <= '0;
            isqrt_ge         <= 1'b0;
            isqrt_step       <= '0;
            x_r              <= '0;
            y_r              <= '0;
            g_r              <= '0;
            o_r              <= '0;
            prod_r           <= '0;
            mul_a            <= '0;
            mul_b            <= '0;
            mul64            <= '0;
            lut_addr         <= '0;
            gelu_passthru    <= 1'b0;
            gelu_zero        <= 1'b0;
        end else begin
            // Restoring divider: MSB of div_num into rem each cycle.
            if (div_busy) begin
                div_trial = {div_rem, div_num[31]};
                div_num   <= {div_num[30:0], 1'b0};
                if (div_trial >= {1'b0, div_denom}) begin
                    div_rem           <= div_trial[31:0] - div_denom;
                    div_quot[div_bit] <= 1'b1;
                end else begin
                    div_rem           <= div_trial[31:0];
                    div_quot[div_bit] <= 1'b0;
                end
                if (div_bit == 6'd0) div_busy <= 1'b0;
                else div_bit <= div_bit - 1'b1;
            end

            case (state)
                ST_IDLE: begin
                    busy_r <= 1'b0;
                    if (start) begin
                        if (len == 8'd0 || len > MAX_LEN[7:0])
                            len_r <= MAX_LEN[IDX_W:0];
                        else
                            len_r <= {1'b0, len[IDX_W-1:0]};
                        param_r <= param;
                        idx     <= '0;
                        acc64   <= '0;
                        sum_exp <= '0;
                        busy_r  <= 1'b1;
                        done_r  <= 1'b0;
                        max_v   <= x_mem[0];
                        unique case (opcode)
                            OP_RESIDUAL: state <= ST_RES_LOAD;
                            OP_GELU:     state <= ST_GELU_LOAD;
                            OP_RMSNORM:  state <= ST_RMS_LOAD;
                            OP_SOFTMAX:  state <= ST_SM_MAX_LOAD;
                            default: begin
                                busy_r           <= 1'b0;
                                done_r           <= 1'b1;
                                complete_count_r <= complete_count_r + 1'b1;
                                state            <= ST_IDLE;
                            end
                        endcase
                    end
                end

                ST_RES_LOAD: begin
                    x_r   <= x_mem[idx[IDX_W-1:0]];
                    y_r   <= y_mem[idx[IDX_W-1:0]];
                    state <= ST_RES;
                end

                ST_RES: begin
                    out_mem[idx[IDX_W-1:0]] <= x_r + y_r;
                    if (idx + 1'b1 >= len_r) state <= ST_DONE;
                    else begin
                        idx   <= idx + 1'b1;
                        state <= ST_RES_LOAD;
                    end
                end

                // GELU: load → index → LUT write
                ST_GELU_LOAD: begin
                    x_r   <= x_mem[idx[IDX_W-1:0]];
                    state <= ST_GELU_ADDR;
                end

                ST_GELU_ADDR: begin
                    gelu_passthru <= (x_r >= FOUR_Q12);
                    gelu_zero     <= (x_r <= -FOUR_Q12);
                    lut_addr      <= gelu_index(x_r);
                    state         <= ST_GELU_WR;
                end

                ST_GELU_WR: begin
                    if (gelu_passthru)
                        out_mem[idx[IDX_W-1:0]] <= x_r;
                    else if (gelu_zero)
                        out_mem[idx[IDX_W-1:0]] <= '0;
                    else
                        out_mem[idx[IDX_W-1:0]] <= gelu_lut[lut_addr];
                    if (idx + 1'b1 >= len_r) state <= ST_DONE;
                    else begin
                        idx   <= idx + 1'b1;
                        state <= ST_GELU_LOAD;
                    end
                end

                ST_RMS_LOAD: begin
                    x_r   <= x_mem[idx[IDX_W-1:0]];
                    state <= ST_RMS_SUM;
                end

                ST_RMS_SUM: begin
                    mul_a <= x_r;
                    mul_b <= x_r;
                    state <= ST_RMS_SUM2;
                end

                ST_RMS_SUM2: begin
                    mul64 <= mul_a * mul_b;
                    state <= ST_RMS_ACC;
                end

                ST_RMS_ACC: begin
                    acc64 <= acc64 + mul64;
                    if (idx + 1'b1 >= len_r) state <= ST_RMS_MEAN;
                    else begin
                        idx   <= idx + 1'b1;
                        state <= ST_RMS_LOAD;
                    end
                end

                ST_RMS_MEAN: begin
                    unique case (len_r)
                        (IDX_W+1)'(1):  mean_u <= acc64 + ({{32{param_r[31]}}, param_r} <<< Q12);
                        (IDX_W+1)'(2):  mean_u <= (acc64 >> 1) + ({{32{param_r[31]}}, param_r} <<< Q12);
                        (IDX_W+1)'(4):  mean_u <= (acc64 >> 2) + ({{32{param_r[31]}}, param_r} <<< Q12);
                        (IDX_W+1)'(8):  mean_u <= (acc64 >> 3) + ({{32{param_r[31]}}, param_r} <<< Q12);
                        default:        mean_u <= (acc64 >> 4) + ({{32{param_r[31]}}, param_r} <<< Q12);
                    endcase
                    isqrt_lo   <= 32'd0;
                    isqrt_hi   <= 32'h0001_0000;
                    isqrt_step <= 6'd0;
                    state      <= ST_ISQRT_MID;
                end

                ST_ISQRT_MID: begin
                    isqrt_mid <= isqrt_lo + ((isqrt_hi - isqrt_lo) >> 1);
                    state     <= ST_ISQRT_SQ;
                end

                ST_ISQRT_SQ: begin
                    isqrt_sq <= isqrt_mid * isqrt_mid;
                    state    <= ST_ISQRT_UPD;
                end

                ST_ISQRT_UPD: begin
                    isqrt_ge = (isqrt_sq <= mean_u);
                    if (isqrt_ge) isqrt_lo <= isqrt_mid;
                    else if (isqrt_mid != 0) isqrt_hi <= isqrt_mid - 1'b1;
                    if (isqrt_step == 6'd15) state <= ST_INV_LAUNCH;
                    else begin
                        isqrt_step <= isqrt_step + 1'b1;
                        state      <= ST_ISQRT_MID;
                    end
                end

                ST_INV_LAUNCH: begin
                    div_num   <= ONE_Q12 * ONE_Q12; // 2^24 fits in 32 bits
                    div_denom <= (isqrt_lo == 32'd0) ? 32'd1 : isqrt_lo;
                    div_quot  <= '0;
                    div_rem   <= '0;
                    div_bit   <= 6'd31;
                    div_busy  <= 1'b1;
                    state     <= ST_INV_WAIT;
                end

                ST_INV_WAIT: begin
                    if (!div_busy) begin
                        inv_rms_q12 <= div_quot;
                        idx         <= '0;
                        state       <= ST_RMS_MUL1;
                    end
                end

                ST_RMS_MUL1: begin
                    x_r   <= x_mem[idx[IDX_W-1:0]];
                    g_r   <= gamma_mem[idx[IDX_W-1:0]];
                    state <= ST_RMS_MUL2;
                end

                ST_RMS_MUL2: begin
                    mul_a <= x_r;
                    mul_b <= inv_rms_q12;
                    state <= ST_RMS_MUL2B;
                end

                ST_RMS_MUL2B: begin
                    mul64  <= mul_a * mul_b;
                    state  <= ST_RMS_MUL3;
                end

                ST_RMS_MUL3: begin
                    prod_r <= mul64 >>> Q12; // Q12
                    mul_a  <= mul64 >>> Q12;
                    mul_b  <= g_r;
                    state  <= ST_RMS_MUL3B;
                end

                ST_RMS_MUL3B: begin
                    mul64 <= mul_a * mul_b;
                    state <= ST_RMS_MUL3C;
                end

                ST_RMS_MUL3C: begin
                    o_r   <= mul64 >>> Q12; // Q12
                    state <= ST_RMS_WR;
                end

                ST_RMS_WR: begin
                    out_mem[idx[IDX_W-1:0]] <= o_r;
                    if (idx + 1'b1 >= len_r) state <= ST_DONE;
                    else begin
                        idx   <= idx + 1'b1;
                        state <= ST_RMS_MUL1;
                    end
                end

                ST_SM_MAX_LOAD: begin
                    x_r   <= x_mem[idx[IDX_W-1:0]];
                    state <= ST_SM_MAX;
                end

                ST_SM_MAX: begin
                    if (x_r > max_v) max_v <= x_r;
                    if (idx + 1'b1 >= len_r) begin
                        idx   <= '0;
                        state <= ST_SM_LOAD;
                    end else begin
                        idx   <= idx + 1'b1;
                        state <= ST_SM_MAX_LOAD;
                    end
                end

                // Softmax exp: load (x-max) → clamp → index → LUT + accumulate
                ST_SM_LOAD: begin
                    x_r   <= x_mem[idx[IDX_W-1:0]] - max_v;
                    state <= ST_SM_CLAMP;
                end

                ST_SM_CLAMP: begin
                    if (x_r > 0) x_r <= '0;
                    else if (x_r < -EIGHT_Q12) x_r <= -EIGHT_Q12;
                    state <= ST_SM_ADDR;
                end

                ST_SM_ADDR: begin
                    lut_addr <= exp_index(x_r);
                    state    <= ST_SM_ACC;
                end

                ST_SM_ACC: begin
                    o_r <= exp_lut[lut_addr];
                    out_mem[idx[IDX_W-1:0]] <= exp_lut[lut_addr];
                    sum_exp <= sum_exp + $unsigned(exp_lut[lut_addr]);
                    if (idx + 1'b1 >= len_r) begin
                        idx   <= '0;
                        state <= ST_SM_LAUNCH;
                    end else begin
                        idx   <= idx + 1'b1;
                        state <= ST_SM_LOAD;
                    end
                end

                ST_SM_LAUNCH: begin
                    // quot = (exp << 16) / sum_exp  (32-bit dividend)
                    div_num   <= $unsigned(out_mem[idx[IDX_W-1:0]]) << 16;
                    div_denom <= (sum_exp == 32'd0) ? 32'd1 : sum_exp;
                    div_quot  <= '0;
                    div_rem   <= '0;
                    div_bit   <= 6'd31;
                    div_busy  <= 1'b1;
                    state     <= ST_SM_WAIT;
                end

                ST_SM_WAIT: begin
                    if (!div_busy) begin
                        out_mem[idx[IDX_W-1:0]] <= div_quot;
                        if (idx + 1'b1 >= len_r) state <= ST_DONE;
                        else begin
                            idx   <= idx + 1'b1;
                            state <= ST_SM_LAUNCH;
                        end
                    end
                end

                ST_DONE: begin
                    busy_r           <= 1'b0;
                    done_r           <= 1'b1;
                    complete_count_r <= complete_count_r + 1'b1;
                    state            <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
