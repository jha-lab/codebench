// ==============================================================
// File generated by Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC
// Version: 2016.4
// Copyright (C) 1986-2017 Xilinx, Inc. All Rights Reserved.
// 
// ==============================================================


`timescale 1 ns / 1 ps

module im2col_cpu_add_32g8j_AddSubnS_3(clk, reset, ce, a, b, s);

// ---- input/output ports list here ----
input clk;
input reset;
input ce;
input [32 - 1 : 0] a;
input [32 - 1 : 0] b;
output [32 - 1 : 0] s;

// ---- register and wire type variables list here ----

// wire for the primary inputs
wire [32 - 1 : 0] a_reg;
wire [32 - 1 : 0] b_reg;

// wires for each small adder
wire [4 - 1 : 0] a0_cb;
wire [4 - 1 : 0] b0_cb;
wire [8 - 1 : 4] a1_cb;
wire [8 - 1 : 4] b1_cb;
wire [12 - 1 : 8] a2_cb;
wire [12 - 1 : 8] b2_cb;
wire [16 - 1 : 12] a3_cb;
wire [16 - 1 : 12] b3_cb;
wire [20 - 1 : 16] a4_cb;
wire [20 - 1 : 16] b4_cb;
wire [24 - 1 : 20] a5_cb;
wire [24 - 1 : 20] b5_cb;
wire [28 - 1 : 24] a6_cb;
wire [28 - 1 : 24] b6_cb;
wire [32 - 1 : 28] a7_cb;
wire [32 - 1 : 28] b7_cb;

// registers for input register array
reg [4 - 1 : 0] a1_cb_regi1[1 - 1 : 0]; 
reg [4 - 1 : 0] b1_cb_regi1[1 - 1 : 0]; 
reg [4 - 1 : 0] a2_cb_regi2[2 - 1 : 0]; 
reg [4 - 1 : 0] b2_cb_regi2[2 - 1 : 0]; 
reg [4 - 1 : 0] a3_cb_regi3[3 - 1 : 0]; 
reg [4 - 1 : 0] b3_cb_regi3[3 - 1 : 0]; 
reg [4 - 1 : 0] a4_cb_regi4[4 - 1 : 0]; 
reg [4 - 1 : 0] b4_cb_regi4[4 - 1 : 0]; 
reg [4 - 1 : 0] a5_cb_regi5[5 - 1 : 0]; 
reg [4 - 1 : 0] b5_cb_regi5[5 - 1 : 0]; 
reg [4 - 1 : 0] a6_cb_regi6[6 - 1 : 0]; 
reg [4 - 1 : 0] b6_cb_regi6[6 - 1 : 0]; 
reg [4 - 1 : 0] a7_cb_regi7[7 - 1 : 0];
reg [4 - 1 : 0] b7_cb_regi7[7 - 1 : 0];

// wires for each full adder sum
wire [32 - 1 : 0] fas;

// wires and register for carry out bit
wire faccout_ini;
wire faccout0_co0; 
wire faccout1_co1; 
wire faccout2_co2; 
wire faccout3_co3; 
wire faccout4_co4; 
wire faccout5_co5; 
wire faccout6_co6; 
wire faccout7_co7;

reg faccout0_co0_reg; 
reg faccout1_co1_reg; 
reg faccout2_co2_reg; 
reg faccout3_co3_reg; 
reg faccout4_co4_reg; 
reg faccout5_co5_reg; 
reg faccout6_co6_reg; 

// registers for output register array
reg [4 - 1 : 0] s0_ca_rego0[6 - 0 : 0]; 
reg [4 - 1 : 0] s1_ca_rego1[6 - 1 : 0]; 
reg [4 - 1 : 0] s2_ca_rego2[6 - 2 : 0]; 
reg [4 - 1 : 0] s3_ca_rego3[6 - 3 : 0]; 
reg [4 - 1 : 0] s4_ca_rego4[6 - 4 : 0]; 
reg [4 - 1 : 0] s5_ca_rego5[6 - 5 : 0]; 
reg [4 - 1 : 0] s6_ca_rego6[6 - 6 : 0]; 

// wire for the temporary output
wire [32 - 1 : 0] s_tmp;

// ---- RTL code for assignment statements/always blocks/module instantiations here ----
assign a_reg = a;
assign b_reg = b;

// small adder input assigments
assign a0_cb = a_reg[4 - 1 : 0];
assign b0_cb = b_reg[4 - 1 : 0];
assign a1_cb = a_reg[8 - 1 : 4];
assign b1_cb = b_reg[8 - 1 : 4];
assign a2_cb = a_reg[12 - 1 : 8];
assign b2_cb = b_reg[12 - 1 : 8];
assign a3_cb = a_reg[16 - 1 : 12];
assign b3_cb = b_reg[16 - 1 : 12];
assign a4_cb = a_reg[20 - 1 : 16];
assign b4_cb = b_reg[20 - 1 : 16];
assign a5_cb = a_reg[24 - 1 : 20];
assign b5_cb = b_reg[24 - 1 : 20];
assign a6_cb = a_reg[28 - 1 : 24];
assign b6_cb = b_reg[28 - 1 : 24];
assign a7_cb = a_reg[32 - 1 : 28];
assign b7_cb = b_reg[32 - 1 : 28];

// input register array
always @ (posedge clk) begin
    if (ce) begin
        a1_cb_regi1 [0] <= a1_cb;
        b1_cb_regi1 [0] <= b1_cb;
        a2_cb_regi2 [0] <= a2_cb;
        b2_cb_regi2 [0] <= b2_cb;
        a3_cb_regi3 [0] <= a3_cb;
        b3_cb_regi3 [0] <= b3_cb;
        a4_cb_regi4 [0] <= a4_cb;
        b4_cb_regi4 [0] <= b4_cb;
        a5_cb_regi5 [0] <= a5_cb;
        b5_cb_regi5 [0] <= b5_cb;
        a6_cb_regi6 [0] <= a6_cb;
        b6_cb_regi6 [0] <= b6_cb;
        a7_cb_regi7 [0] <= a7_cb;
        b7_cb_regi7 [0] <= b7_cb;
        a2_cb_regi2 [1] <= a2_cb_regi2 [0];
        b2_cb_regi2 [1] <= b2_cb_regi2 [0];
        a3_cb_regi3 [1] <= a3_cb_regi3 [0];
        b3_cb_regi3 [1] <= b3_cb_regi3 [0];
        a4_cb_regi4 [1] <= a4_cb_regi4 [0];
        b4_cb_regi4 [1] <= b4_cb_regi4 [0];
        a5_cb_regi5 [1] <= a5_cb_regi5 [0];
        b5_cb_regi5 [1] <= b5_cb_regi5 [0];
        a6_cb_regi6 [1] <= a6_cb_regi6 [0];
        b6_cb_regi6 [1] <= b6_cb_regi6 [0];
        a7_cb_regi7 [1] <= a7_cb_regi7 [0];
        b7_cb_regi7 [1] <= b7_cb_regi7 [0];
        a3_cb_regi3 [2] <= a3_cb_regi3 [1];
        b3_cb_regi3 [2] <= b3_cb_regi3 [1];
        a4_cb_regi4 [2] <= a4_cb_regi4 [1];
        b4_cb_regi4 [2] <= b4_cb_regi4 [1];
        a5_cb_regi5 [2] <= a5_cb_regi5 [1];
        b5_cb_regi5 [2] <= b5_cb_regi5 [1];
        a6_cb_regi6 [2] <= a6_cb_regi6 [1];
        b6_cb_regi6 [2] <= b6_cb_regi6 [1];
        a7_cb_regi7 [2] <= a7_cb_regi7 [1];
        b7_cb_regi7 [2] <= b7_cb_regi7 [1];
        a4_cb_regi4 [3] <= a4_cb_regi4 [2];
        b4_cb_regi4 [3] <= b4_cb_regi4 [2];
        a5_cb_regi5 [3] <= a5_cb_regi5 [2];
        b5_cb_regi5 [3] <= b5_cb_regi5 [2];
        a6_cb_regi6 [3] <= a6_cb_regi6 [2];
        b6_cb_regi6 [3] <= b6_cb_regi6 [2];
        a7_cb_regi7 [3] <= a7_cb_regi7 [2];
        b7_cb_regi7 [3] <= b7_cb_regi7 [2];
        a5_cb_regi5 [4] <= a5_cb_regi5 [3];
        b5_cb_regi5 [4] <= b5_cb_regi5 [3];
        a6_cb_regi6 [4] <= a6_cb_regi6 [3];
        b6_cb_regi6 [4] <= b6_cb_regi6 [3];
        a7_cb_regi7 [4] <= a7_cb_regi7 [3];
        b7_cb_regi7 [4] <= b7_cb_regi7 [3];
        a6_cb_regi6 [5] <= a6_cb_regi6 [4];
        b6_cb_regi6 [5] <= b6_cb_regi6 [4];
        a7_cb_regi7 [5] <= a7_cb_regi7 [4];
        b7_cb_regi7 [5] <= b7_cb_regi7 [4];
        a7_cb_regi7 [6] <= a7_cb_regi7 [5];
        b7_cb_regi7 [6] <= b7_cb_regi7 [5];
    end
end

// carry out bit processing
always @ (posedge clk) begin
    if (ce) begin
        faccout0_co0_reg <= faccout0_co0;
        faccout1_co1_reg <= faccout1_co1;
        faccout2_co2_reg <= faccout2_co2;
        faccout3_co3_reg <= faccout3_co3;
        faccout4_co4_reg <= faccout4_co4;
        faccout5_co5_reg <= faccout5_co5;
        faccout6_co6_reg <= faccout6_co6;
    end
end

// small adder generation 
        im2col_cpu_add_32g8j_AddSubnS_3_fadder u0 (
            .faa    ( a0_cb ),
            .fab    ( b0_cb ),
            .facin  ( faccout_ini ),
            .fas    ( fas[3:0] ),
            .facout ( faccout0_co0 )
        );
        im2col_cpu_add_32g8j_AddSubnS_3_fadder u1 (
            .faa    ( a1_cb_regi1[0] ),
            .fab    ( b1_cb_regi1[0] ),
            .facin  ( faccout0_co0_reg),
            .fas    ( fas[7:4] ),
            .facout ( faccout1_co1 )
        );
        im2col_cpu_add_32g8j_AddSubnS_3_fadder u2 (
            .faa    ( a2_cb_regi2[1] ),
            .fab    ( b2_cb_regi2[1] ),
            .facin  ( faccout1_co1_reg),
            .fas    ( fas[11:8] ),
            .facout ( faccout2_co2 )
        );
        im2col_cpu_add_32g8j_AddSubnS_3_fadder u3 (
            .faa    ( a3_cb_regi3[2] ),
            .fab    ( b3_cb_regi3[2] ),
            .facin  ( faccout2_co2_reg),
            .fas    ( fas[15:12] ),
            .facout ( faccout3_co3 )
        );
        im2col_cpu_add_32g8j_AddSubnS_3_fadder u4 (
            .faa    ( a4_cb_regi4[3] ),
            .fab    ( b4_cb_regi4[3] ),
            .facin  ( faccout3_co3_reg),
            .fas    ( fas[19:16] ),
            .facout ( faccout4_co4 )
        );
        im2col_cpu_add_32g8j_AddSubnS_3_fadder u5 (
            .faa    ( a5_cb_regi5[4] ),
            .fab    ( b5_cb_regi5[4] ),
            .facin  ( faccout4_co4_reg),
            .fas    ( fas[23:20] ),
            .facout ( faccout5_co5 )
        );
        im2col_cpu_add_32g8j_AddSubnS_3_fadder u6 (
            .faa    ( a6_cb_regi6[5] ),
            .fab    ( b6_cb_regi6[5] ),
            .facin  ( faccout5_co5_reg),
            .fas    ( fas[27:24] ),
            .facout ( faccout6_co6 )
        );
        im2col_cpu_add_32g8j_AddSubnS_3_fadder_f u7 (
            .faa    ( a7_cb_regi7[6] ),
            .fab    ( b7_cb_regi7[6] ),
            .facin  ( faccout6_co6_reg ),
            .fas    ( fas[31 :28] ),
            .facout ( faccout7_co7 )
        );

assign faccout_ini = 1'b0;

// output register array
always @ (posedge clk) begin
    if (ce) begin
        s0_ca_rego0 [0] <= fas[4-1 : 0];
        s1_ca_rego1 [0] <= fas[8-1 : 4];
        s2_ca_rego2 [0] <= fas[12-1 : 8];
        s3_ca_rego3 [0] <= fas[16-1 : 12];
        s4_ca_rego4 [0] <= fas[20-1 : 16];
        s5_ca_rego5 [0] <= fas[24-1 : 20];
        s6_ca_rego6 [0] <= fas[28-1 : 24];
        s0_ca_rego0 [1] <= s0_ca_rego0 [0];
        s0_ca_rego0 [2] <= s0_ca_rego0 [1];
        s0_ca_rego0 [3] <= s0_ca_rego0 [2];
        s0_ca_rego0 [4] <= s0_ca_rego0 [3];
        s0_ca_rego0 [5] <= s0_ca_rego0 [4];
        s0_ca_rego0 [6] <= s0_ca_rego0 [5];
        s1_ca_rego1 [1] <= s1_ca_rego1 [0];
        s1_ca_rego1 [2] <= s1_ca_rego1 [1];
        s1_ca_rego1 [3] <= s1_ca_rego1 [2];
        s1_ca_rego1 [4] <= s1_ca_rego1 [3];
        s1_ca_rego1 [5] <= s1_ca_rego1 [4];
        s2_ca_rego2 [1] <= s2_ca_rego2 [0];
        s2_ca_rego2 [2] <= s2_ca_rego2 [1];
        s2_ca_rego2 [3] <= s2_ca_rego2 [2];
        s2_ca_rego2 [4] <= s2_ca_rego2 [3];
        s3_ca_rego3 [1] <= s3_ca_rego3 [0];
        s3_ca_rego3 [2] <= s3_ca_rego3 [1];
        s3_ca_rego3 [3] <= s3_ca_rego3 [2];
        s4_ca_rego4 [1] <= s4_ca_rego4 [0];
        s4_ca_rego4 [2] <= s4_ca_rego4 [1];
        s5_ca_rego5 [1] <= s5_ca_rego5 [0];
    end
end

// get the s_tmp, assign it to the primary output
assign s_tmp[4-1 : 0] = s0_ca_rego0[6];
assign s_tmp[8-1 : 4] = s1_ca_rego1[5];
assign s_tmp[12-1 : 8] = s2_ca_rego2[4];
assign s_tmp[16-1 : 12] = s3_ca_rego3[3];
assign s_tmp[20-1 : 16] = s4_ca_rego4[2];
assign s_tmp[24-1 : 20] = s5_ca_rego5[1];
assign s_tmp[28-1 : 24] = s6_ca_rego6[0];
assign s_tmp[32 - 1 : 28] = fas[31 :28];

assign s = s_tmp;

endmodule

// short adder
module im2col_cpu_add_32g8j_AddSubnS_3_fadder 
#(parameter
    N = 4
)(
    input  [N-1 : 0]  faa,
    input  [N-1 : 0]  fab,
    input  wire  facin,
    output [N-1 : 0]  fas,
    output wire  facout
);
assign {facout, fas} = faa + fab + facin;

endmodule

// the final stage short adder
module im2col_cpu_add_32g8j_AddSubnS_3_fadder_f 
#(parameter
    N = 4
)(
    input  [N-1 : 0]  faa,
    input  [N-1 : 0]  fab,
    input  wire  facin,
    output [N-1 : 0]  fas,
    output wire  facout
);
assign {facout, fas} = faa + fab + facin;

endmodule

`timescale 1 ns / 1 ps
module im2col_cpu_add_32g8j(
    clk,
    reset,
    ce,
    din0,
    din1,
    dout);

parameter ID = 32'd1;
parameter NUM_STAGE = 32'd1;
parameter din0_WIDTH = 32'd1;
parameter din1_WIDTH = 32'd1;
parameter dout_WIDTH = 32'd1;
input clk;
input reset;
input ce;
input[din0_WIDTH - 1:0] din0;
input[din1_WIDTH - 1:0] din1;
output[dout_WIDTH - 1:0] dout;



im2col_cpu_add_32g8j_AddSubnS_3 im2col_cpu_add_32g8j_AddSubnS_3_U(
    .clk( clk ),
    .reset( reset ),
    .ce( ce ),
    .a( din0 ),
    .b( din1 ),
    .s( dout ));

endmodule

