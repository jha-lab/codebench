/*****************************************************************************
 *                                McPAT/CACTI
 *                      SOFTWARE LICENSE AGREEMENT
 *            Copyright 2012 Hewlett-Packard Development Company, L.P.
 *                          All Rights Reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.”
 *
 ***************************************************************************/

#ifndef __CONST_H__
#define __CONST_H__

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/*  The following are things you might want to change
 *  when compiling
 */
 
#define techNode        14
#if(techNode == 14)
#define delayModel		1  // 0 for default horowitz delay model, 1 for 14nm library based delay
#define nocModel		1  // 0 for default built in mcpat noc model, 1 for orion noc model
#define Mono_type		0  /* Monolithic implementaion type. 0 is for block-level monolithic, 1 is for transistor-level monolithic,  and 2 is for gate-monolithic. */

#define CapGate		   	(0.1327e-15)  
#define CapDrain	   	(0.0402e-15)
 
#define CapInINV 		(0.3743e-15)  
#define CapInNAND       (0.1272e-15)  
#define CapInNOR     	(0.6408e-15)  
#define CapInDFF     	(0.1272e-15)
#define CapOutINV       (0.1270e-15)  
#define CapOutNAND      (0.1653e-15)  
#define CapOutNOR     	(0.5022e-15)  
#define CapOutDFF     	(0.4077e-15)

const float logicGateWidth[8][5]={		// [type][size] types: 0:NOT, 1:NAND2, 2:NAND3, 3:NAND4, 4:NOR2, 5:NOR3, 6:NOR4, 7:DFF. Sizes: 1-16x except DFF which is always 1x
						{0.126, 0.126, 0.196, 0.336, 0.616},
						{0.196, 0.196, 0.196, 0.336, 0.616},
						{0.518, 0.518, 0.588, 1.008, 1.848},
						{0.840, 0.840, 0.980, 1.680, 3.080},
						{0.196, 0.336, 0.616, 1.726, 2.296},
						{0.518, 0.798, 1.428, 2.688, 5.208},
						{0.840, 1.260, 2.240, 4.200, 8.120},
						{1.092, 1.092, 1.092, 1.092, 1.092},};					

const float coefNOTRise[5][10] = {
						{0.41899837221, 0.14467028056, 0.91338972102, -0.00092100963, 0.00830068631, -0.31342853802, 0.16118571765, 2.19430489312, 0.00005856246, -0.00400644685},
						{0.36514219042, 0.11792383277, 0.91912806076, -0.00091204650, 0.01556816965, -0.07996504415, 0.13534459868, 2.12274791943, 0.00006849260, 0.00303417438},
						{0.24405365910, 0.09725216036, 0.97918383895, -0.00084970219, 0.02414921417, -0.07287623326, 0.13068573026, 2.07880455710, -0.00013663843, 0.01075899961},
						{0.13315908382, 0.07671442702, 1.10314392252, -0.00067765675, 0.03347418709, -0.08519250587, 0.12214939701, 2.07426463270, -0.00017324618, 0.01961070342},
						{0.00543206861, 0.05999040929, 1.30179498265, -0.00050984055, 0.04290470307, -0.22253114400, 0.11390440647, 2.21318416484, -0.00008671859, 0.02558726757},};

const float coefNOTFall[5][10] = {
						{0.67904190256, 0.05586135221, 0.89548768127, -0.00081457152, 0.00754797126, -0.15691100154, 0.12730770024, 2.17647867767, 0.00048628795, -0.00139175127},
						{0.57517518498, 0.03427674996, 0.91296557156, -0.00084329495, 0.01356023857, -0.10760598821, 0.12716601804, 2.13756903176, 0.00020842802, 0.00222702689},
						{0.46979885184, 0.01389560382, 0.97127557114, -0.00074546128, 0.02092788775, -0.12197565669, 0.12661118001, 2.10731235809, -0.00000989034, 0.00804305290},
						{0.38506643177, -0.00484319751, 1.07896560180, -0.00058011400, 0.02928669498, -0.14219167374, 0.11926485959, 2.11121593371, -0.00002102764, 0.01499943663},
						{0.31224186195, -0.01965632599, 1.22915893276, -0.00043039023, 0.03827096419, -0.15332650426, 0.10908502915, 2.15258232288, 0.00006710129, 0.02271056500},};

const float coefNANDRise[5][10] = {
						{0.99846955138, 0.20601127106, 0.90866287143, -0.00057819352, 0.00216204011, -0.28487433373, 0.12510380160, 2.20033634489, 0.00096585416, -0.00299890404},
						{0.94798678812, 0.19079879288, 0.90798808777, -0.00071782693, 0.00515895501, -0.17173284377, 0.14182156893, 2.17271297636, 0.00053129169, -0.00274037451},
						{0.88526717750, 0.18126142665, 0.90917525937, -0.00086480824, 0.00931969841, -0.07588939179, 0.14916235333, 2.12265526345, 0.00017550961, -0.00004683666},
						{0.79885388821, 0.17348417315, 0.93331106018, -0.00091778158, 0.01338781564, -0.01510678017, 0.15374299303, 2.05370982587, -0.00023167003, 0.00615781691},
						{0.72315602335, 0.16565053215, 0.97796356136, -0.00089022592, 0.01674317237, -0.08280626438, 0.15337949098, 2.05028453976, -0.00041102681, 0.01183125130},};
						
						
const float coefNANDFall[5][10] = {
						{0.69775422667, -0.00426622339, 0.90086254061, -0.00035816562, 0.00208767352, -0.36658653959, 0.11492685946, 2.20805759347, 0.00075216650, -0.00289330058},
						{0.58080736453, -0.02030479203, 0.91159432116, -0.00042444521, 0.00473009078, -0.25058729718, 0.12843831657, 2.18985137690, 0.00060343986, -0.00391337480},
						{0.46667223535, -0.03255361049, 0.93148929589, -0.00047750625, 0.00851234009, -0.36618773933, 0.15985815211, 2.15176108394, 0.00008778999, -0.00405542706},
						{0.34345093423, -0.04494416783, 0.98059137552, -0.00045571955, 0.01318168609, -0.08916803266, 0.15410178169, 2.09305873223, 0.00012322859, -0.00383566505},
						{0.20720634201, -0.05367861960, 1.06489168566, -0.00043168688, 0.01772782475, -0.02062189880, 0.15963943386, 2.02703782768, 0.00000486256, -0.00305349095},};	
											
const float coefNORRise[5][10] = {
						{0.32175952720, 0.06323716714, 0.94775964628, -0.00050342597, 0.00545899471, -0.39747413688, 0.15530676570, 2.21245584189, 0.00043787272, -0.00474929290},
						{0.19296670704, 0.04575193920, 0.97068156973, -0.00052885409, 0.00983117676, -0.24750541231, 0.16838109553, 2.16584187040, 0.00024766136, -0.00526198442},
						{0.02765321837, 0.02967553624, 1.02681392498, -0.00050114180, 0.01499358433, -0.13488203103, 0.18226282427, 2.10052541852, 0.00000430897, -0.00561416510},
						{-0.18733524873, 0.01659603127, 1.13346012604, -0.00043527582, 0.01956608360, -0.01818861471, 0.19093860204, 2.03454829067, -0.00010535250, -0.00688712052},
						{-0.40065646169, 0.00680591417, 1.26326371419, -0.00035958367, 0.02281900600, 0.16036256884, 0.19142824415, 1.93556816882, -0.00006902002, -0.00797971296},};

const float coefNORFall[5][10] = {
						{1.81033688495, 0.10963512753, 0.91139418509, -0.00061949028, 0.00419524069, -0.37362201369, 0.11510241805, 2.20531690781, 0.00083624226, -0.00223276490},
						{1.73902278528, 0.10175375175, 0.91638066579, -0.00074693553, 0.00711098363, -0.22970658698, 0.11770873969, 2.16152425682, 0.00058936413, -0.00021147560},
						{1.65056769155, 0.09499970518, 0.93729367085, -0.00079596402, 0.00991887557, -0.11630081870, 0.11820014062, 2.10193004091, 0.00031851875, 0.00385413008},
						{1.57049366998, 0.08821686182, 0.97005957791, -0.00078357001, 0.01235301680, -0.06028944509, 0.11606583846, 2.05941132251, 0.00015263062, 0.00812520361},
						{1.50716676111, 0.08309873120, 1.00402488902, -0.00075194059, 0.01395574648, 0.01309587122, 0.10974282740, 2.01794052750, 0.00010945761, 0.01211429582},};	

const float tempCoefNAND[9] =
				{-0.000186167043965, 1.05742880972, -0.00042748555792, 1.13173852235, 0.0021574852572, 0.356357797629, -9.82716783217e-05, 0.102571860839, -21.8289916531 };					
const float tempCoefNOT[9] =
				{-6.89995071464e-05, 1.02457188942, -0.000685569285084, 1.19673786408, 0.00195601004289, 0.416782090177, -9.76924918415e-05, 0.102658429958, -21.906278447 };
const float tempCoefNOR[9] =
				{-0.000675709891, 1.26946107784, -1.6094409046e-05, 1.04157554698, 0.00163255320278, 0.494205243326, -2.63080536131e-05, 0.0502445753147, -12.2395880857 };
const float tempCoefDFF[9] =
				{-0.000186167043965, 1.05742880972, 0.000995989304813, 0.698181818182, 0.00176071055381, 0.477659352142, -9.87403617716e-05, 0.104100389726, -22.2430761555 };
#endif

/*
 * Address bits in a word, and number of output bits from the cache
 */

/*
was: #define ADDRESS_BITS 32
now: I'm using 42 bits as in the Power4,
since that's bigger then the 36 bits on the Pentium 4
and 40 bits on the Opteron
*/
const int ADDRESS_BITS = 42;

/*dt: In addition to the tag bits, the tags also include 1 valid bit, 1 dirty bit, 2 bits for a 4-state
  cache coherency protocoll (MESI), 1 bit for MRU (change this to log(ways) for full LRU).
  So in total we have 1 + 1 + 2 + 1 = 5 */
const int EXTRA_TAG_BITS = 5;

/* limits on the various N parameters */

const unsigned int MAXDATAN     = 512;      // maximum for Ndwl and Ndbl
const unsigned int MAXSUBARRAYS = 1048576;  // maximum subarrays for data and tag arrays
const unsigned int MAXDATASPD   = 256;      // maximum for Nspd
const unsigned int MAX_COL_MUX  = 256;



#define ROUTER_TYPES 3
#define WIRE_TYPES 6

const double Cpolywire = 0;


/* Threshold voltages (as a proportion of Vdd)
   If you don't know them, set all values to 0.5 */
#define VTHFA1         0.452
#define VTHFA2         0.304
#define VTHFA3         0.420
#define VTHFA4         0.413
#define VTHFA5         0.405
#define VTHFA6         0.452
#define VSINV          0.452
#define VTHCOMPINV     0.437
#define VTHMUXNAND     0.548  // TODO : this constant must be revisited
#define VTHEVALINV     0.452
#define VTHSENSEEXTDRV 0.438


//WmuxdrvNANDn and WmuxdrvNANDp are no longer being used but it's part of the old
//delay_comparator function which we are using exactly as it used to be, so just setting these to 0
const double WmuxdrvNANDn = 0;
const double WmuxdrvNANDp = 0;


/*===================================================================*/
/*
 * The following are things you probably wouldn't want to change.
 */

#define BIGNUM 1e30
#define INF 9999999
#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))

/* Used to communicate with the horowitz model */
#define RISE 1
#define FALL 0
#define NCH  1
#define PCH  0


#define EPSILON 0.5 //v4.1: This constant is being used in order to fix floating point -> integer
//conversion problems that were occuring within CACTI. Typical problem that was occuring was
//that with different compilers a floating point number like 3.0 would get represented as either
//2.9999....or 3.00000001 and then the integer part of the floating point number (3.0) would
//be computed differently depending on the compiler. What we are doing now is to replace
//int (x) with (int) (x+EPSILON) where EPSILON is 0.5. This would fix such problems. Note that
//this works only when x is an integer >= 0.
/*
 * Sheng thinks this is more a solution to solve the simple truncate problem
 * (http://www.cs.tut.fi/~jkorpela/round.html) rather than the problem mentioned above.
 * Unfortunately, this solution causes nasty bugs (different results when using O0 and O3).
 * Moreover, round is not correct in CACTI since when an extra fraction of bit/line is needed,
 * we need to provide a complete bit/line even the fraction is just 0.01.
 * So, in later version than 6.5 we use (int)ceil() to get double to int conversion.
 */

#define EPSILON2 0.1
#define EPSILON3 0.6


#define MINSUBARRAYROWS 16 //For simplicity in modeling, for the row decoding structure, we assume
//that each row predecode block is composed of at least one 2-4 decoder. When the outputs from the
//row predecode blocks are combined this means that there are at least 4*4=16 row decode outputs
#define MAXSUBARRAYROWS 262144 //Each row predecode block produces a max of 2^9 outputs. So
//the maximum number of row decode outputs will be 2^9*2^9
#define MINSUBARRAYCOLS 2
#define MAXSUBARRAYCOLS 262144


#define INV 0
#define NOR 1
#define NAND 2
#define DFF 3


#define NUMBER_TECH_FLAVORS 4

#define NUMBER_INTERCONNECT_PROJECTION_TYPES 2 //aggressive and conservative
//0 = Aggressive projections, 1 = Conservative projections
#define NUMBER_WIRE_TYPES 4 //local, semi-global and global
//1 = 'Semi-global' wire type, 2 = 'Global' wire type


const int dram_cell_tech_flavor = 3;


#define VBITSENSEMIN 0.08 //minimum bitline sense voltage is fixed to be 80 mV.

#define fopt 4.0

#define INPUT_WIRE_TO_INPUT_GATE_CAP_RATIO 0
#define BUFFER_SEPARATION_LENGTH_MULTIPLIER 1
#define NUMBER_MATS_PER_REDUNDANT_MAT 8

#define NUMBER_STACKED_DIE_LAYERS 1

// this variable can be set to carry out solution optimization for
// a maximum area allocation.
#define STACKED_DIE_LAYER_ALLOTED_AREA_mm2 0 //6.24 //6.21//71.5

// this variable can also be employed when solution optimization
// with maximum area allocation is carried out.
#define MAX_PERCENT_AWAY_FROM_ALLOTED_AREA 50

// this variable can also be employed when solution optimization
// with maximum area allocation is carried out.
#define MIN_AREA_EFFICIENCY 20

// this variable can be employed when solution with a desired
// aspect ratio is required.
#define STACKED_DIE_LAYER_ASPECT_RATIO 1

// this variable can be employed when solution with a desired
// aspect ratio is required.
#define MAX_PERCENT_AWAY_FROM_ASPECT_RATIO 101

// this variable can be employed to carry out solution optimization
// for a certain target random cycle time.
#define TARGET_CYCLE_TIME_ns 1000000000

#define NUMBER_PIPELINE_STAGES 4

// this can be used to model the length of interconnect
// between a bank and a crossbar
#define LENGTH_INTERCONNECT_FROM_BANK_TO_CROSSBAR 0 //3791 // 2880//micron

#define IS_CROSSBAR 0
#define NUMBER_INPUT_PORTS_CROSSBAR 8
#define NUMBER_OUTPUT_PORTS_CROSSBAR 8
#define NUMBER_SIGNALS_PER_PORT_CROSSBAR 256


#define MAT_LEAKAGE_REDUCTION_DUE_TO_SLEEP_TRANSISTORS_FACTOR 1
#define LEAKAGE_REDUCTION_DUE_TO_LONG_CHANNEL_HP_TRANSISTORS_FACTOR 1

#define PAGE_MODE 0

#define MAIN_MEM_PER_CHIP_STANDBY_CURRENT_mA 60
// We are actually not using this variable in the CACTI code. We just want to acknowledge that
// this current should be multiplied by the DDR(n) system VDD value to compute the standby power
// consumed during precharge.


const double VDD_STORAGE_LOSS_FRACTION_WORST = 0.125;
const double CU_RESISTIVITY = 0.022; //ohm-micron
const double BULK_CU_RESISTIVITY = 0.018; //ohm-micron
const double PERMITTIVITY_FREE_SPACE = 8.854e-18; //F/micron

const static uint32_t sram_num_cells_wl_stitching_ = 16;
const static uint32_t dram_num_cells_wl_stitching_ = 64;
const static uint32_t comm_dram_num_cells_wl_stitching_ = 256;
const static double num_bits_per_ecc_b_          = 8.0;

const double    bit_to_byte  = 8.0;

#define MAX_NUMBER_GATES_STAGE 20
#define MAX_NUMBER_HTREE_NODES 20
#define NAND2_LEAK_STACK_FACTOR 0.2
#define NAND3_LEAK_STACK_FACTOR 0.2
#define NOR2_LEAK_STACK_FACTOR 0.2
#define INV_LEAK_STACK_FACTOR  0.5
#define MAX_NUMBER_ARRAY_PARTITIONS 1000000

// abbreviations used in this project
// ----------------------------------
//
//  num  : number
//  rw   : read/write
//  rd   : read
//  wr   : write
//  se   : single-ended
//  sz   : size
//  F    : feature
//  w    : width
//  h    : height or horizontal
//  v    : vertical or velocity


enum ram_cell_tech_type_num
{
  itrs_hp   = 0,
  itrs_lstp = 1,
  itrs_lop  = 2,
  lp_dram   = 3,
  comm_dram = 4
};

const double pppm[4]      = {1,1,1,1};
const double pppm_lkg[4]  = {0,1,1,0};
const double pppm_dyn[4]  = {1,0,0,0};
const double pppm_Isub[4] = {0,1,0,0};
const double pppm_Ig[4]   = {0,0,1,0};
const double pppm_sc[4]   = {0,0,0,1};

const double Ilinear_to_Isat_ratio =2.0;



#endif
