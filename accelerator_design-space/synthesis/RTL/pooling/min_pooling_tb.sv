module min_pooling_tb();

parameter IL = 4, FL = 16;
parameter size = 4;
parameter width = $clog2(size);

logic clk;
logic [IL+FL-1:0] im [size-1:0];
logic en;
logic input_ready;

logic [IL+FL-1:0] om;
logic done;

integer i;

min_pooling	min_pooling_0
(
	.clk		(clk),
	.im		(im),
	.en		(en),
	.input_ready	(input_ready),
	.om		(om),
	.done		(done)
);

always_ff @(posedge clk) begin
	$display("en=%b input_ready=%b done=%b", en, input_ready, done);
	for (i = 0; i < size; i = i + 1) begin
		$display("im[%d]=%d", i, im[i]);
	end
	$display("om=%d", om);
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
	en = 1;
	#10
	im[0] = 520;
	im[1] = 360;
	im[2] = 1378;
	im[3] = 280;
	input_ready = 1;
	#10
	input_ready = 0;
	#500

	$finish;
end

endmodule
