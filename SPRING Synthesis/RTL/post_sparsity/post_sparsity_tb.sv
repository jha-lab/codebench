module post_sparsity_tb();

parameter IL = 4, FL = 16;
parameter length = 32;

logic clk, reset;
logic signed [IL+FL-1:0] i_im [15:0];
logic [length-1:0] i_mask;
logic input_ready;
logic output_taken;
logic signed [IL+FL-1:0] o_im [15:0];
logic [length-1:0] o_mask;
logic [1:0] state;

integer i,j;

post_sparsity	post_sparsity_0
(
	.clk		(clk),
	.reset		(reset),
	.i_im		(i_im),
	.i_mask		(i_mask),
	.input_ready	(input_ready),
	.output_taken	(output_taken),
	.o_im		(o_im),
	.o_mask		(o_mask),
	.state		(state)
);

always_ff @(posedge clk) begin
	$display("reset=%b ", reset,
		 "input_ready=%b output_taken=%b\n ", input_ready, output_taken,
		 "i_mask=%b\n o_mask=%b\n ", i_mask, o_mask,
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

	i_mask = 32'b01001001110101011011010011011011;
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
