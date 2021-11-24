module backward_tb();

parameter IL = 4, FL = 16;
parameter size = 16;

logic clk, reset;
logic signed [IL+FL-1:0] dout [size-1:0];
logic [4:0] num;
logic signed [IL+FL-1:0] batch [size-1:0];
logic signed [IL+FL-1:0] norm [size-1:0];
logic signed [IL+FL-1:0] mu;
logic signed [IL+FL-1:0] vari;
logic signed [IL+FL-1:0] gamma;
logic input_ready;
logic output_taken;

logic signed [IL+FL-1:0] dX;
logic signed [IL+FL-1:0] dgamma;
logic signed [IL+FL-1:0] dbeta;
logic [1:0] state;
logic done;

integer i,j;

backward		backward_0
(
	.clk		(clk),
	.reset		(reset),
	.dout		(dout),
        .num		(num),
        .batch		(batch),
        .norm		(norm),
        .mu		(mu),
        .vari		(vari),
        .gamma		(gamma),
        .input_ready	(input_ready),
        .output_taken	(output_taken),
        .dX		(dX),
        .dgamma		(dgamma),
        .dbeta		(dbeta),
        .state		(state),
        .done		(done)	
);

always_ff @(posedge clk) begin
	$display("reset=%b state=%b done=%b input_ready=%b output_taken=%B ", reset, state, done, input_ready, output_taken,
		 "num=%d mu=%d vari=%d gamma=%d dgamma=%d dbeta=%d", num, mu, vari, gamma, dgamma, dbeta);
	for (i = 0; i < size; i = i + 1) begin
		$display("dout[%d]=%d batch[%d]=%d norm[%d]=%d dX[%d]=%d", i, dout[i], i, batch[i], i, norm[i], i, dX[i]);
	end
end

always #5 clk = !clk;

initial begin
	clk = 0;
	reset = 1;
	#10
	reset = 0;
	input_ready = 1;
	output_taken = 0;
	num = 10;
	mu = 5;
	vari = 3;
	gamma = 2;
	for (j = 0; j < size; j = j + 1) begin
		dout[j] = j;
		batch[j] = 2 * j + 1;
		norm[j] = 3 * j + 2;
	end
	#10
	input_ready = 0;
	#250
	output_taken = 1;
	#10
	output_taken = 0;
	#10

	$finish;
end

endmodule
