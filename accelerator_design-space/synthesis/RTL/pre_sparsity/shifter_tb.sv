module shifter_tb();

parameter IL = 4, FL = 16;

logic clk, reset;
logic signed [IL+FL-1:0] i_0, i_1, i_2, i_3, i_4, i_5, i_6, i_7;
logic signed [IL+FL-1:0] i_8, i_9, i_10, i_11, i_12, i_13, i_14, i_15;
logic signed [IL+FL-1:0] w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7;
logic signed [IL+FL-1:0] w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15;
logic [1:0] state;
logic input_taken;
logic signed [IL+FL-1:0] oi_0, oi_1, oi_2, oi_3, oi_4, oi_5, oi_6, oi_7;
logic signed [IL+FL-1:0] oi_8, oi_9, oi_10, oi_11, oi_12, oi_13, oi_14, oi_15;
logic signed [IL+FL-1:0] ow_0, ow_1, ow_2, ow_3, ow_4, ow_5, ow_6, ow_7;
logic signed [IL+FL-1:0] ow_8, ow_9, ow_10, ow_11, ow_12, ow_13, ow_14, ow_15;
logic output_ready;

shifter shifter_0
(
	.clk		(clk),
	.reset		(reset),
	.i_0		(i_0),
	.i_1		(i_1),
	.i_2		(i_2),
	.i_3		(i_3),
	.i_4		(i_4),
	.i_5		(i_5),
	.i_6		(i_6),
	.i_7		(i_7),
	.i_8		(i_8),
	.i_9		(i_9),
	.i_10		(i_10),
	.i_11		(i_11),
	.i_12		(i_12),
	.i_13		(i_13),
	.i_14		(i_14),
	.i_15		(i_15),
	.w_0		(w_0),
	.w_1		(w_1),
	.w_2		(w_2),
	.w_3		(w_3),
	.w_4		(w_4),
	.w_5		(w_5),
	.w_6		(w_6),
	.w_7		(w_7),
	.w_8		(w_8),
	.w_9		(w_9),
	.w_10		(w_10),
	.w_11		(w_11),
	.w_12		(w_12),
	.w_13		(w_13),
	.w_14		(w_14),
	.w_15		(w_15),	
	.state		(state),
	.input_taken	(input_taken),
	.oi_0		(oi_0),
	.oi_1		(oi_1),	
	.oi_2		(oi_2),
	.oi_3		(oi_3),
	.oi_4		(oi_4),
	.oi_5		(oi_5),
	.oi_6		(oi_6),
	.oi_7		(oi_7),
	.oi_8		(oi_8),
	.oi_9		(oi_9),
	.oi_10		(oi_10),
	.oi_11		(oi_11),
	.oi_12		(oi_12),
	.oi_13		(oi_13),
	.oi_14		(oi_14),
	.oi_15		(oi_15),
	.ow_0		(ow_0),
	.ow_1		(ow_1),
	.ow_2		(ow_2),
	.ow_3		(ow_3),
	.ow_4		(ow_4),
	.ow_5		(ow_5),
	.ow_6		(ow_6),
	.ow_7		(ow_7),
	.ow_8		(ow_8),
	.ow_9		(ow_9),
	.ow_10		(ow_10),
	.ow_11		(ow_11),
	.ow_12		(ow_12),
	.ow_13		(ow_13),
	.ow_14		(ow_14),
	.ow_15		(ow_15),
	.output_ready	(output_ready)
);

always_ff @(posedge clk) begin
	$display("clk=%b, reset=%b ", clk, reset,
		 "i_0=%b, i_1=%b, i_2=%b, i_3=%b ", i_0, i_1, i_2, i_3,
		 "i_4=%b, i_5=%b, i_6=%b, i_7=%b ", i_4, i_5, i_6, i_7,
		 "i_8=%b, i_9=%b, i_10=%b, i_11=%b ", i_8, i_9, i_10, i_11,
		 "i_12=%b, i_13=%b, i_14=%b, i_15=%b ", i_12, i_13, i_14, i_15,
		 "w_0=%b, w_1=%b, w_2=%b, w_3=%b ", w_0, w_1, w_2, w_3,
		 "w_4=%b, w_5=%b, w_6=%b, w_7=%b ", w_4, w_5, w_6, w_7,
		 "w_8=%b, w_9=%b, w_10=%b, w_11=%b ", w_8, w_9, w_10, w_11,
		 "w_12=%b, w_13=%b, w_14=%b, w_15=%b ", w_12, w_13, w_14, w_15,
		 "state=%b, input_taken=%b ", state, input_taken,
		 "oi_0=%b, oi_1=%b, oi_2=%b, oi_3=%b ", oi_0, oi_1, oi_2, oi_3,
		 "oi_4=%b, oi_5=%b, oi_6=%b, oi_7=%b ", oi_4, oi_5, oi_6, oi_7,
		 "oi_8=%b, oi_9=%b, oi_10=%b, oi_11=%b ", oi_8, oi_9, oi_10, oi_11,
		 "oi_12=%b, oi_13=%b, oi_14=%b, oi_15=%b ", oi_12, oi_13, oi_14, oi_15,
		 "ow_0=%b, ow_1=%b, ow_2=%b, ow_3=%b ", ow_0, ow_1, ow_2, ow_3,
		 "ow_4=%b, ow_5=%b, ow_6=%b, ow_7=%b ", ow_4, ow_5, ow_6, ow_7,
		 "ow_8=%b, ow_9=%b, ow_10=%b, ow_11=%b ", ow_8, ow_9, ow_10, ow_11,
		 "ow_12=%b, ow_13=%b, ow_14=%b, ow_15=%b ", ow_12, ow_13, ow_14, ow_15,
		 "output_ready=%b\n ", output_ready);
end

initial begin
	forever begin
		clk = 0;
		#5
		clk = 1;
		#5
		clk = 0;
	end
end

initial begin
	reset = 1;
	#10
	reset = 0;
	#10
        state = 2'b10;
	i_0 = 20'b1111<<10;
	i_1 = 0;
	i_2 = 20'b11101<<10;
	i_3 = 20'b1011<<10;
	i_4 = 0;
	i_5 = 20'b1111<<10;
        i_6 = 0;
        i_7 = 0;
        i_8 = 20'b1011<<10;
        i_9 = 20'b10111<<10;
	i_10 = 20'b1111<<10;
        i_11 = 0;
        i_12 = 20'b11101<<10;
        i_13 = 0;
        i_14 = 0;
	i_15 = 20'b101<<10;
	w_0 = 20'b1111<<10;
        w_1 = 0;
        w_2 = 0;
        w_3 = 20'b1011<<10;
        w_4 = 20'b10111<<10;
        w_5 = 0;
        w_6 = 0;
        w_7 = 20'b11101<<10;
        w_8 = 20'b1011<<10;
        w_9 = 0;
        w_10 = 20'b1111<<10;
        w_11 = 20'b11111<<10;
        w_12 = 0;
        w_13 = 20'b1011<<10;
        w_14 = 0;
        w_15 = 20'b101<<10;
	#200
	state = 2'b00;
	#500

	$finish;
end

endmodule
