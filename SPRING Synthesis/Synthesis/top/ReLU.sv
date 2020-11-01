module ReLU
(
	clk,		//i
	reset,		//i
	i,		//i
	f		//o
);

parameter IL = 8, FL = 12;

input clk, reset;
input signed [IL+FL-1:0] i;
output logic signed [IL+FL-1:0] f;

always_comb begin
	f = (i[IL+FL-1] == 0) ? i : {(IL+FL){1'b0}};
end

endmodule
