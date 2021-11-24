module scalar_tb();

parameter IL = 4, FL = 16;
parameter size = 16;

logic clk, reset;
logic [1:0] mode;
logic input_ready, output_taken;
logic signed [IL+FL-1:0] [size-1:0] in1;
logic signed [IL+FL-1:0] [size-1:0] in2;

logic [1:0] state;
logic signed [IL+FL-1:0] [size-1:0] out;

scalar	#(.IL(IL), .FL(FL), .size(size))	scalar_0
(
	.clk		(clk),
	.reset		(reset),
	.mode		(mode),
	.input_ready	(input_ready),
	.output_taken	(output_taken),
	.in1		(in1),
	.in2		(in2),
	.state		(state),
	.out		(out)
);

int j;
always_ff @(posedge clk) begin
        $display("reset=%b mode=%b input_ready=%b output_taken=%b state=%b", reset, mode, input_ready, output_taken, state);
	for (j = 0; j < size; j++) begin        
		$display("in1[%d]=%d", j, in1[j]);
		$display("in2[%d]=%d", j, in2[j]);
        	$display("out[%d]=%d", j, out[j]);
	end
end

always #5 clk = !clk;

int i;
initial begin
        $dumpfile("scalar_tb.vcd");
        $dumpvars;
        clk = 0;
	reset = 1;
	#10
	reset = 0;
	input_ready = 1;
	for (i = 0; i < size; i++) begin
		in1[i] = 2*i + 1;
		in2[i] = i;
	end
	mode = 2'b00;
	#10
	input_ready = 0;
	#50
	output_taken = 1;
	#10
	output_taken = 0;
	mode = 2'b01;
	input_ready = 1;
	#10
	input_ready = 0;
	#30
	output_taken = 1;
	#10
	output_taken = 0;
	#10
	mode = 2'b10;
	input_ready = 1;
	#10
	input_ready = 0;
	#50
	output_taken = 1;
	#10
	output_taken = 0;
	#10
	mode = 2'b11;
	input_ready = 1;
	#10
	input_ready = 0;
	#50
	output_taken = 1;
	#10
	output_taken = 0;
        #10

        $finish;
end


endmodule
