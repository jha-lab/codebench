module L2_tb();

parameter IL = 4, FL = 16;
parameter size = 16;
parameter width = $clog2(size);

logic signed [IL+FL-1:0] yHat [size-1:0];
logic signed [IL+FL-1:0] y [size-1:0];
logic [width-1:0] num;
logic reset;
logic en;

logic signed [IL+FL-1:0] sum;

logic clk;

integer i,j;

L2 #(.IL(IL), .FL(FL), .size(size)) L2_0
(
	.reset		(reset),
	.en		(en),
	.yHat		(yHat),
	.y		(y),
	.num		(num),
	.sum		(sum)
);

always_ff @(posedge clk) begin
        $display("reset=%b en=%b num=%d sum=%d", reset, en, num, sum);
        for (i = 0; i < size; i = i + 1) begin
                $display("yHat[%d]=%d", i, yHat[i]);
                $display("y[%d]=%d", i, y[i]);
        end
end

always #5 clk = !clk;

initial begin
        $dumpfile("L2_tb.vcd");
        $dumpvars;
        clk = 0;
	en = 1;
        reset = 1;
        num = 10;
	for (j = 0; j < num; j = j + 1) begin
                yHat[j] = j;
                y[j] = 3;
        end
        #10
        reset = 0;
        #50

        $finish;
end



endmodule
