#include <string.h> // memset, memcpy
#include <math.h> //tanhf

#include "nn.h"

// unconventional: include generated c-file in here
#include "generated_weights.c"

static float temp1[32];
static float temp2[32];

static float deepset_sum_neighbor[8];
static float deepset_sum_obstacle[8];

static float relu(float num) {
	if (num > 0) {
		return num;
	} else {
		return 0;
	}
}

static void layer(int rows, int cols, const float in[], const float layer_weight[][cols], const float layer_bias[],
			float * output, int use_activation) {
	for(int ii = 0; ii < cols; ii++) {
		output[ii] = 0;
		for (int jj = 0; jj < rows; jj++) {
			output[ii] += in[jj] * layer_weight[jj][ii];
		}
		output[ii] += layer_bias[ii];
		if (use_activation == 1) {
			output[ii] = relu(output[ii]);
		}
	}
}

static const float* n_phi(const float input[]) {
	layer(2, 32, input, weights_n_phi.layers_0_weight, weights_n_phi.layers_0_bias, temp1, 1);
	layer(32, 32, temp1, weights_n_phi.layers_1_weight, weights_n_phi.layers_1_bias, temp2, 1);
	layer(32, 8, temp2, weights_n_phi.layers_2_weight, weights_n_phi.layers_2_bias, temp1, 0);

	return temp1;
}

static const float* n_rho(const float input[]) {
	layer(8, 32, input, weights_n_rho.layers_0_weight, weights_n_rho.layers_0_bias, temp1, 1);
	layer(32, 32, temp1, weights_n_rho.layers_1_weight, weights_n_rho.layers_1_bias, temp2, 1);
	layer(32, 8, temp2, weights_n_rho.layers_2_weight, weights_n_rho.layers_2_bias, temp1, 0);

	return temp1;
}

static const float* o_phi(const float input[]) {
	layer(2, 32, input, weights_o_phi.layers_0_weight, weights_o_phi.layers_0_bias, temp1, 1);
	layer(32, 32, temp1, weights_o_phi.layers_1_weight, weights_o_phi.layers_1_bias, temp2, 1);
	layer(32, 8, temp2, weights_o_phi.layers_2_weight, weights_o_phi.layers_2_bias, temp1, 0);

	return temp1;
}

static const float* o_rho(const float input[]) {
	layer(8, 32, input, weights_o_rho.layers_0_weight, weights_o_rho.layers_0_bias, temp1, 1);
	layer(32, 32, temp1, weights_o_rho.layers_1_weight, weights_o_rho.layers_1_bias, temp2, 1);
	layer(32, 8, temp2, weights_o_rho.layers_2_weight, weights_o_rho.layers_2_bias, temp1, 0);

	return temp1;
}

static const float* psi(const float input[]) {
	layer(18, 32, input, weights_psi.layers_0_weight, weights_psi.layers_0_bias, temp1, 1);
	layer(32, 32, temp1, weights_psi.layers_1_weight, weights_psi.layers_1_bias, temp2, 1);
	layer(32, 2, temp2, weights_psi.layers_2_weight, weights_psi.layers_2_bias, temp1, 0);

	// // scaling part
	// const float psi_min = -0.5;
	// const float psi_max = 0.5;
	// for (int i = 0; i < 2; ++i) {
	// 	temp1[i] = (tanhf(temp1[i]) + 1.0f) / 2.0f * ((psi_max - psi_min) + psi_min);
	// }

	return temp1;
}

void nn_reset(void)
{
	memset(deepset_sum_neighbor, 0, sizeof(deepset_sum_neighbor));
	memset(deepset_sum_obstacle, 0, sizeof(deepset_sum_obstacle));
}

void nn_add_neighbor(const float input[2])
{
	const float* phi = n_phi(input);
	// sum result
	for (int i = 0; i < 8; ++i) {
		deepset_sum_neighbor[i] += phi[i];
	}
}

void nn_add_obstacle(const float input[2])
{
	const float* phi = o_phi(input);
	// sum result
	for (int i = 0; i < 8; ++i) {
		deepset_sum_obstacle[i] += phi[i];
	}
}

const float* nn_eval(const float goal[2])
{
	static float pi_input[18];

	const float* neighbors = n_rho(deepset_sum_neighbor);
	memcpy(&pi_input[0], neighbors, 8 * sizeof(float));

	const float* obstacles = o_rho(deepset_sum_obstacle);
	memcpy(&pi_input[8], obstacles, 8 * sizeof(float));

	memcpy(&pi_input[16], goal, 2 * sizeof(float));

	return psi(pi_input);
}

