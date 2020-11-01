module mac
(
	clk, 		//i
  	reset, 		//i
	en_sum,		//i
  	a,		//i
  	b,		//i
	part_sum	//i
  	f		//o
);

	parameter IL = 4, FL = 16;
	input clk, reset, en_sum;
	input signed [IL+FL-1:0] a, b;
	input signed [(IL+FL)*2-1:0] part_sum;
	output logic signed [(IL+FL)*2-1:0] f;

	logic signed [IL+FL-1:0] reg_a, reg_b;
	logic signed [(IL+FL)*2-1:0] mul, reg_mul, add, reg_add;

	always_ff @(posedge clk) begin
		if (reset == 1) begin
			reg_a <= 0;
			reg_b <= 0;
			reg_mul <= 0;
			if (en_sum == 1) begin
				reg_add <= part_sum;
			end
			else begin
				reg_add <= 0;
			end
		end
		else begin
			reg_a <= a;
			reg_b <= b;
			reg_mul <= mul;
			reg_add <= add;
		end
	end

	always_comb begin
		mul = reg_a * reg_b;
		add = reg_mul + reg_add;
	end

	always_comb begin
		f = reg_add;
	end
endmodule
