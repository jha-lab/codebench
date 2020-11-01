module update_mask_tb();

parameter IL = 4, FL = 16;
parameter length = 32;
parameter p_length = $clog2(length);

logic clk, reset;
logic [length-1:0] i_mask;
logic signed [IL+FL-1:0] out [15:0];
logic input_ready;
logic output_taken;
logic [length-1:0] o_mask;
logic [1:0] state;

integer i;

update_mask update_mask_0
(
	.clk		(clk),
        .reset		(reset),
        .i_mask		(i_mask),
        .out		(out),
        .input_ready	(input_ready),
        .output_taken	(output_taken),
        .o_mask		(o_mask),
        .state		(state)	
);

always_ff @(posedge clk) begin
	$display("reset=%b ", reset,
		 "i_mask=%b ", i_mask,
		 "input_ready=%b output_taken=%b ", input_ready, output_taken,
		 "o_mask=%b state=%b ", o_mask, state);
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
	reset = 0;
	#10
	reset = 1;
	#10
	reset = 0;
	
	i_mask = 32'b01001001110101011011010011011011;
	for (i = 0; i < 16; i = i + 1) begin
		out[i] = i+1;
	end
	out[5] = 0;
	out[7] = 0;
	out[8] = 0;
	out[11] = 0;
	out[15] = 0;
	input_ready = 1;
	output_taken = 0;
	#10
	input_ready = 0;
	#1000

	$finish;
end

endmodule
