#include <string.h> // memset, memcpy
#include <math.h> //tanhf
#include <stdbool.h>

// Debug
#include <stdio.h>

#include "nn.h"

// unconventional: include generated c-file in here
#include "generated_weights.c"

static float temp1[32];
static float temp2[32];

static float deepset_sum_neighbor[8];
static float deepset_sum_obstacle[8];

static float closest_dist;
static float closest[2];
static float min_distance;

// static const float b_gamma = 0.1;
// static const float b_exph = 1.0;
static const float robot_radius = 0.15; // m
static const float max_v = 0.5; // m/s
static const float pi_max = 0.9f * 0.5f;

// Barrier stuff
static float barrier_grad_phi[2];
static bool barrier_alpha_condition;
static const float deltaR = 0.5 * 0.05;
static const float Rsense = 3.0;
static const float barrier_gamma = 0.005;

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

static float clip(float value, float min, float max) {
	if (value < min) {
		return min;
	}
	if (value > max) {
		return max;
	}
	return value;
}

// static void barrier(float x, float y, float D, float* vx, float *vy) {
// 	float normP = sqrtf(x*x + y*y);
// 	float H = normP - D;

// 	float factor = powf(H, -b_exph) / normP;

// 	*vx = b_gamma * factor * x;
// 	*vy = b_gamma * factor * y;
// }

// static void APF(float* vel)
// {
// 	if (isfinite(closest_dist)) {
// 		float vx, vy;
// 		barrier(closest[0], closest[1], min_distance, &vx, &vy);
// 		vel[0] -= vx;
// 		vel[1] -= vy;
// 	}
// }

void nn_reset()
{
	memset(deepset_sum_neighbor, 0, sizeof(deepset_sum_neighbor));
	memset(deepset_sum_obstacle, 0, sizeof(deepset_sum_obstacle));

	memset(barrier_grad_phi, 0, sizeof(barrier_grad_phi));
	barrier_alpha_condition = false;
}

void nn_add_neighbor(const float input[2])
{
	const float* phi = n_phi(input);
	// sum result
	for (int i = 0; i < 8; ++i) {
		deepset_sum_neighbor[i] += phi[i];
	}

	// barrier
	const float Rsafe = robot_radius;
	const float dist = sqrtf(powf(input[0], 2) + powf(input[1], 2)) - robot_radius;

	if (dist > Rsafe) {
		const float denominator = dist * (dist - Rsafe) / (Rsense - Rsafe);
		// compute vector to the closest contact point
		float x = input[0] * (1 - robot_radius / (dist+robot_radius));
		float y = input[1] * (1 - robot_radius / (dist+robot_radius));
		printf("P N: %f,%f\n", x, y);

		barrier_grad_phi[0] += x / denominator;
		barrier_grad_phi[1] += y / denominator;

		if (dist < Rsafe + deltaR) {
			barrier_alpha_condition = true;
		}
	}
}

void nn_add_obstacle(const float input[2])
{
	const float* phi = o_phi(input);
	// sum result
	for (int i = 0; i < 8; ++i) {
		deepset_sum_obstacle[i] += phi[i];
	}

	// barrier
	float closest_x = clip(0, input[0] - 0.5f, input[0] + 0.5f);
	float closest_y = clip(0, input[1] - 0.5f, input[1] + 0.5f);

	const float Rsafe = robot_radius;
	const float dist = sqrtf(powf(closest_x, 2) + powf(closest_y, 2));

	if (dist > Rsafe) {
		const float denominator = dist * (dist - Rsafe) / (Rsense - Rsafe);
		printf("P O: %f,%f\n",closest_x, closest_y);
		barrier_grad_phi[0] += closest_x / denominator;
		barrier_grad_phi[1] += closest_y / denominator;

		if (dist < Rsafe + deltaR) {
			barrier_alpha_condition = true;
		}
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

	const float* empty = psi(pi_input);

	printf("empty: %f %f\n", empty[0], empty[1]);

	// // fun hack: goToGoal policy
	// const float kp = 1.0;
	// temp1[0] = kp*goal[0];
	// temp1[1] = kp*goal[1];

	// for barrier, we need to scale empty to pi_max:
	float empty_norm = sqrtf(powf(empty[0], 2) + powf(empty[1], 2));
	if (empty_norm > 0) {
		const float scale = fminf(1.0f, pi_max / empty_norm);
		temp1[0] = scale * empty[0];
		temp1[1] = scale * empty[1];
	} else {
		temp1[0] = empty[0];
		temp1[1] = empty[1];
	}

	// Barrier:
	float b[2] = {0, 0};
	b[0] = -barrier_gamma * barrier_grad_phi[0];
	b[1] = -barrier_gamma * barrier_grad_phi[1];

	printf("barrier: %f %f\n", b[0], b[1]);

	float alpha = 1.0;
	if (barrier_alpha_condition) {
		float bnorm = sqrtf(powf(b[0], 2) + powf(b[1], 2));
		float pinorm = sqrtf(powf(temp1[0], 2) + powf(temp1[1], 2));
		if (pinorm > 0) {
			alpha = fminf(1.0f, bnorm / pinorm);
		}
	}

	temp1[0] = b[0] + alpha * temp1[0];
	temp1[1] = b[1] + alpha * temp1[1];

	// scaling
	float temp1_norm = sqrtf(powf(temp1[0], 2) + powf(temp1[1], 2));
	float alpha2 = 1.0;
	if (temp1_norm > 0) {
		alpha2 = fminf(1.0f, max_v / temp1_norm);
	}
	temp1[0] *= alpha2;
	temp1[1] *= alpha2;

	// APF(temp1);

	// float inv_alpha = fmaxf(fabsf(temp1[0]), fabsf(temp1[1])) / max_v;
	// inv_alpha = fmaxf(inv_alpha, 1.0);
	// temp1[0] /= inv_alpha;
	// temp1[1] /= inv_alpha;

	return temp1;
}

