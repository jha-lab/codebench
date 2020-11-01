module update_output_tb();

parameter IL = 4, FL = 16;

logic clk, reset;
logic signed [IL+FL-1:0] i_im [15:0];
logic input_ready;
logic output_taken;
logic signed [IL+FL-1:0] o_im [15:0];
logic [1:0] state;

integer i,j;

update_output update_output_0
(
	.clk		(clk),
	.reset		(reset),
	.i_im		(i_im),
	.input_ready	(input_ready),
	.output_taken	(output_taken),
	.o_im		(o_im),
	.state		(state)
);

always_ff @(posedge clk) begin
	$display("reset=%b ", reset,
		 "input_ready=%b ", input_ready,
		 "state=%b ", state);
	for (i = 0; i < 16; i = i + 1) begin
		$display("i_im[%d]", i, "=%b ", i_im[i]);
	end
	for (i = 0; i < 16; i = i + 1) begin
                $display("o_im[%d]", i, "=%b ", o_im[i]);
        end
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

	for (j = 0; j < 16; j = j + 1) begin
                i_im[j] = j+1;
        end
        i_im[5] = 0;
        i_im[7] = 0;
        i_im[8] = 0;
        i_im[11] = 0;
        i_im[15] = 0;
        input_ready = 1;
        output_taken = 0;	
	#10
	input_ready = 0;
	#500
	$finish;
end

endmodule
