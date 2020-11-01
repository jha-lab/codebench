module sqrt_tb();

parameter IL = 4, FL = 16;

logic clk, reset;
logic signed [IL+FL-1:0] in;
logic input_ready;
logic output_taken;

logic signed [IL+FL-1:0] out;
logic [1:0] state;
logic done;

sqrt		sqrt_0
(
	.clk		(clk),
	.reset		(reset),
	.in		(in),
	.input_ready	(input_ready),
	.output_taken	(output_taken),
	.out		(out),
	.state		(state),
	.done		(done)
);

always_ff @(posedge clk) begin
	$display("reset=%b input_ready=%b output_taken=%b ", reset, input_ready, output_taken,
		 "state=%b done=%b ", state, done,
		 "in=%d out=%d", in, out);
end

always #5 clk = !clk;

initial begin
	clk = 0;
	reset = 1;
	#10
	reset = 0;
	input_ready = 1;
	output_taken = 0;
	in = 1020;
	#200
	output_taken = 1;
	#10
	output_taken = 0;
	#10
	$finish;
end

endmodule
