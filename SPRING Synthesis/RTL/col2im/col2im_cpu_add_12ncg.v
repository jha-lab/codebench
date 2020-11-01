// ==============================================================
// File generated by Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC
// Version: 2016.4
// Copyright (C) 1986-2017 Xilinx, Inc. All Rights Reserved.
// 
// ==============================================================


`timescale 1 ns / 1 ps

module col2im_cpu_add_12ncg_AddSubnS_4(clk, reset, ce, a, b, s);

// ---- input/output ports list here ----
input clk;
input reset;
input ce;
input [12 - 1 : 0] a;
input [12 - 1 : 0] b;
output [12 - 1 : 0] s;

// ---- register and wire type variables list here ----

// wire for the primary inputs
wire [12 - 1 : 0] a_reg;
wire [12 - 1 : 0] b_reg;

// wires for each small adder
wire [4 - 1 : 0] a0_cb;
wire [4 - 1 : 0] b0_cb;
wire [8 - 1 : 4] a1_cb;
wire [8 - 1 : 4] b1_cb;
wire [12 - 1 : 8] a2_cb;
wire [12 - 1 : 8] b2_cb;

// registers for input register array
reg [4 - 1 : 0] a1_cb_regi1[1 - 1 : 0]; 
reg [4 - 1 : 0] b1_cb_regi1[1 - 1 : 0]; 
reg [4 - 1 : 0] a2_cb_regi2[2 - 1 : 0];
reg [4 - 1 : 0] b2_cb_regi2[2 - 1 : 0];

// wires for each full adder sum
wire [12 - 1 : 0] fas;

// wires and register for carry out bit
wire faccout_ini;
wire faccout0_co0; 
wire faccout1_co1; 
wire faccout2_co2;

reg faccout0_co0_reg; 
reg faccout1_co1_reg; 

// registers for output register array
reg [4 - 1 : 0] s0_ca_rego0[1 - 0 : 0]; 
reg [4 - 1 : 0] s1_ca_rego1[1 - 1 : 0]; 

// wire for the temporary output
wire [12 - 1 : 0] s_tmp;

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

// input register array
always @ (posedge clk) begin
    if (ce) begin
        a1_cb_regi1 [0] <= a1_cb;
        b1_cb_regi1 [0] <= b1_cb;
        a2_cb_regi2 [0] <= a2_cb;
        b2_cb_regi2 [0] <= b2_cb;
        a2_cb_regi2 [1] <= a2_cb_regi2 [0];
        b2_cb_regi2 [1] <= b2_cb_regi2 [0];
    end
end

// carry out bit processing
always @ (posedge clk) begin
    if (ce) begin
        faccout0_co0_reg <= faccout0_co0;
        faccout1_co1_reg <= faccout1_co1;
    end
end

// small adder generation 
        col2im_cpu_add_12ncg_AddSubnS_4_fadder u0 (
            .faa    ( a0_cb ),
            .fab    ( b0_cb ),
            .facin  ( faccout_ini ),
            .fas    ( fas[3:0] ),
            .facout ( faccout0_co0 )
        );
        col2im_cpu_add_12ncg_AddSubnS_4_fadder u1 (
            .faa    ( a1_cb_regi1[0] ),
            .fab    ( b1_cb_regi1[0] ),
            .facin  ( faccout0_co0_reg),
            .fas    ( fas[7:4] ),
            .facout ( faccout1_co1 )
        );
        col2im_cpu_add_12ncg_AddSubnS_4_fadder_f u2 (
            .faa    ( a2_cb_regi2[1] ),
            .fab    ( b2_cb_regi2[1] ),
            .facin  ( faccout1_co1_reg ),
            .fas    ( fas[11 :8] ),
            .facout ( faccout2_co2 )
        );

assign faccout_ini = 1'b0;

// output register array
always @ (posedge clk) begin
    if (ce) begin
        s0_ca_rego0 [0] <= fas[4-1 : 0];
        s1_ca_rego1 [0] <= fas[8-1 : 4];
        s0_ca_rego0 [1] <= s0_ca_rego0 [0];
    end
end

// get the s_tmp, assign it to the primary output
assign s_tmp[4-1 : 0] = s0_ca_rego0[1];
assign s_tmp[8-1 : 4] = s1_ca_rego1[0];
assign s_tmp[12 - 1 : 8] = fas[11 :8];

assign s = s_tmp;

endmodule

// short adder
module col2im_cpu_add_12ncg_AddSubnS_4_fadder 
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
module col2im_cpu_add_12ncg_AddSubnS_4_fadder_f 
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
module col2im_cpu_add_12ncg(
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



col2im_cpu_add_12ncg_AddSubnS_4 col2im_cpu_add_12ncg_AddSubnS_4_U(
    .clk( clk ),
    .reset( reset ),
    .ce( ce ),
    .a( din0 ),
    .b( din1 ),
    .s( dout ));

endmodule

