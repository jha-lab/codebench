module mean_tb();

parameter IL = 4, FL = 16;
parameter size = 16;

logic start;
logic signed [IL+FL-1:0] batch [size-1:0];
logic [4:0] num;

logic signed [IL+FL-1:0] out;

logic clk;
integer i,j;

mean		mean_0
(
	.start		(start),
	.batch		(batch),
	.num		(num),
	.out		(out)	
);

always_ff@(posedge clk) begin
	$display("start=%b num=%d out=%d", start, num, out);
	for (i = 0; i < size; i = i + 1) begin
		$display("batch[%d]=%d", i, batch[i]);
	end
end

always #5 clk = !clk;

initial begin
	clk = 0;
	start = 1;
	for (j = 0; j < size; j = j + 1) begin
		batch[j] = 5*j + 8;
	end
	num = 10;
	#10
	start = 0;
	#20
	$finish;
end

endmodule
