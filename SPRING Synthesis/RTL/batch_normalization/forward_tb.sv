module forward_tb();

parameter IL = 4, FL = 16;
parameter size = 16;

logic clk, reset;
logic signed [IL+FL-1:0] batch [size-1:0];
logic [4:0] num;
logic signed [IL+FL-1:0] gamma;
logic signed [IL+FL-1:0] beta;
logic input_ready;
logic output_taken;

logic signed [IL+FL-1:0] out [size-1:0];
logic signed [IL+FL-1:0] norm [size-1:0];
logic signed [IL+FL-1:0] mu;
logic signed [IL+FL-1:0] vari;
logic [1:0] state;
logic done;

integer i,j;

forward #(.IL(IL), .FL(FL), .size(size))		forward_0
(
	.clk		(clk),
	.reset		(reset),
	.batch		(batch),
	.num		(num),
	.gamma		(gamma),
	.beta		(beta),
	.input_ready	(input_ready),
	.output_taken	(output_taken),
	.out		(out),
	.norm		(norm),
	.mu		(mu),
	.vari		(vari),
	.state		(state),
	.done		(done)
);

always_ff @(posedge clk) begin
	$display("reset=%b input_ready=%b output_taken=%b state=%b done=%b ", reset, input_ready, output_taken, state, done,
		 "num=%d gamma=%d beta=%d mu=%d vari=%d", num, gamma, beta, mu, vari);
	for (i = 0; i < size; i = i + 1) begin
		$display("batch[%d]=%d", i, batch[i]);
		$display("out[%d]=%d", i, out[i]);
		$display("norm[%d]=%d", i, norm[i]);
	end
end

always #5 clk = !clk;

initial begin
	$dumpfile("forward_tb.vcd");
	$dumpvars;
	clk = 0;
	reset = 1;
	#10
	reset = 0;
	input_ready = 1;
	output_taken = 0;
	num = 10;
	gamma = 2;
	beta = 5;
	for (j = 0; j < num; j = j + 1) begin
		batch[j] = j;
	end
	#10
	input_ready = 0;
	#300
	output_taken = 1;
	#10
	output_taken = 0;
	#10

	$finish;
end

endmodule
