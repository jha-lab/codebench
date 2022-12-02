module mask_tb();

parameter length = 32;

logic clk, reset;
logic [length-1:0] i_mask;
logic [length-1:0] w_mask; 
logic input_ready;
logic output_taken;
logic [length-1:0] o_mask;
logic [length-1:0] xor_i_mask;
logic [length-1:0] xor_w_mask;
logic [1:0] state;

mask mask_0
(
	.clk		(clk),
	.reset		(reset),
	.i_mask		(i_mask),
	.w_mask		(w_mask),
	.input_ready	(input_ready),
	.output_taken	(output_taken),
	.o_mask		(o_mask),
	.xor_i_mask	(xor_i_mask),
	.xor_w_mask	(xor_w_mask),
	.state		(state)
);

always_ff @(posedge clk) begin
	$display("clk=%b\n", clk,
		 "reset=%b\n", reset,
		 "i_mask=%b\n", i_mask,
		 "w_mask=%b\n", w_mask,
		 "input_ready=%b\n", input_ready,
		 "output_taken=%b\n", output_taken,
		 "o_mask=%b\n", o_mask,
		 "xor_i_mask=%b\n", xor_i_mask,
		 "xor_w_mask=%b\n", xor_w_mask,
		 "state=%b\n\n", state);
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
	#10
	i_mask = 32'b11010011110100111101001111010011;
        w_mask = 32'b10111001101110001001001100110010;
        input_ready = 1;
        output_taken = 0;
	#10
	input_ready = 0;
	#100

	$finish;
end

endmodule
