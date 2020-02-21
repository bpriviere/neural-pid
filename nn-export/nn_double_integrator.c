#include <string.h> // memset, memcpy
#include <math.h> //tanhf, exp 
#include <stdbool.h>
#include <stdio.h>

#include "nn.h"

// unconventional: include generated c-file in here
#include "generated_weights.c"

static float temp1[64];
static float temp2[64];

static float deepset_sum_neighbor[16];
static float deepset_sum_obstacle[16];

// static const float b_gamma = 0.1;
// static const float b_exph = 1.0;
static const float robot_radius = 0.15; // m
static const float max_v = 0.5; //0.5; // m/s
static const float max_a = 2.0;

// Barrier stuff
static float barrier_grad_phi[2];
static float barrier_grad_phi_dot[2];
static float minp;
// static bool barrier_alpha_condition;
static const float deltaR = 2*(0.5*0.05 + 0.05*0.05/(2*2.0));
static const float Rsense = 3.0;
// static const float barrier_gamma = 0.01;//0.005;
static const float barrier_kp = 0.01;//0.005;
static const float barrier_kv = 1.0;//0.005;
// static const float barrier_kh = 0.5; 
static const float epsilon = 0.01;
static const float pi_max = 0.2;

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
	layer(4, 64, input, weights_n_phi.layers_0_weight, weights_n_phi.layers_0_bias, temp1, 1);
	layer(64, 64, temp1, weights_n_phi.layers_1_weight, weights_n_phi.layers_1_bias, temp2, 1);
	layer(64, 16, temp2, weights_n_phi.layers_2_weight, weights_n_phi.layers_2_bias, temp1, 0);

	return temp1;
}

static const float* n_rho(const float input[]) {
	layer(16, 64, input, weights_n_rho.layers_0_weight, weights_n_rho.layers_0_bias, temp1, 1);
	layer(64, 64, temp1, weights_n_rho.layers_1_weight, weights_n_rho.layers_1_bias, temp2, 1);
	layer(64, 16, temp2, weights_n_rho.layers_2_weight, weights_n_rho.layers_2_bias, temp1, 0);

	return temp1;
}

static const float* o_phi(const float input[]) {
	layer(4, 64, input, weights_o_phi.layers_0_weight, weights_o_phi.layers_0_bias, temp1, 1);
	layer(64, 64, temp1, weights_o_phi.layers_1_weight, weights_o_phi.layers_1_bias, temp2, 1);
	layer(64, 16, temp2, weights_o_phi.layers_2_weight, weights_o_phi.layers_2_bias, temp1, 0);

	return temp1;
}

static const float* o_rho(const float input[]) {
	layer(16, 64, input, weights_o_rho.layers_0_weight, weights_o_rho.layers_0_bias, temp1, 1);
	layer(64, 64, temp1, weights_o_rho.layers_1_weight, weights_o_rho.layers_1_bias, temp2, 1);
	layer(64, 16, temp2, weights_o_rho.layers_2_weight, weights_o_rho.layers_2_bias, temp1, 0);

	return temp1;
}

static const float* psi(const float input[]) {
	layer(16+16+4, 64, input, weights_psi.layers_0_weight, weights_psi.layers_0_bias, temp1, 1);
	layer(64, 64, temp1, weights_psi.layers_1_weight, weights_psi.layers_1_bias, temp2, 1);
	layer(64, 2, temp2, weights_psi.layers_2_weight, weights_psi.layers_2_bias, temp1, 0);

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


void nn_reset()
{
	memset(deepset_sum_neighbor, 0, sizeof(deepset_sum_neighbor));
	memset(deepset_sum_obstacle, 0, sizeof(deepset_sum_obstacle));

	memset(barrier_grad_phi, 0, sizeof(barrier_grad_phi));
	memset(barrier_grad_phi_dot, 0, sizeof(barrier_grad_phi_dot));
	minp = 100.0;
}

void nn_add_neighbor(const float input[4])
{
	const float* phi = n_phi(input);
	// sum result
	for (int i = 0; i < 16; ++i) {
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

		barrier_grad_phi[0] += x / denominator;
		barrier_grad_phi[1] += y / denominator;

		const float ptv = x*input[2] + y*input[3];
		const float d1 = dist*(dist - Rsafe);
		const float d2 = powf(dist,3)*(dist - Rsafe);
		const float d3 = powf(dist,2)*powf(dist - Rsafe,2);

		barrier_grad_phi_dot[0] += input[2]/d1 - x*ptv/d2 - x*ptv/d3;
		barrier_grad_phi_dot[1] += input[3]/d1 - y*ptv/d2 - y*ptv/d3;

		// if (dist < Rsafe + deltaR) {
		// 	barrier_alpha_condition = true;
		// }

		if (dist < minp) {
			minp = dist;
		}
	}
}

void nn_add_obstacle(const float input[4])
{
	const float* phi = o_phi(input);
	// sum result
	for (int i = 0; i < 16; ++i) {
		deepset_sum_obstacle[i] += phi[i];
	}

	// barrier
	float closest_x = clip(0, input[0] - 0.5f, input[0] + 0.5f);
	float closest_y = clip(0, input[1] - 0.5f, input[1] + 0.5f);

	const float Rsafe = robot_radius;
	const float dist = sqrtf(powf(closest_x, 2) + powf(closest_y, 2));

	if (dist > Rsafe) {
		const float denominator = dist * (dist - Rsafe) / (Rsense - Rsafe);
		barrier_grad_phi[0] += closest_x / denominator;
		barrier_grad_phi[1] += closest_y / denominator;

		const float ptv = closest_x*input[2] + closest_y*input[3];
		const float d1 = dist*(dist - Rsafe);
		const float d2 = powf(dist,3)*(dist - Rsafe);
		const float d3 = powf(dist,2)*powf(dist - Rsafe,2);

		barrier_grad_phi_dot[0] += input[2]/d1 - closest_x*ptv/d2 - closest_x*ptv/d3;
		barrier_grad_phi_dot[1] += input[3]/d1 - closest_y*ptv/d2 - closest_y*ptv/d3;

		// if (dist < Rsafe + deltaR) {
		// 	barrier_alpha_condition = true;
		// }

		if (dist < minp) {
			minp = dist;
		}
				
	}
}

static float temp_dbg[10];

const float* nn_eval(const float goal[4])
{
	static float pi_input[36];

	const float* neighbors = n_rho(deepset_sum_neighbor);
	memcpy(&pi_input[0], neighbors, 16 * sizeof(float));

	const float* obstacles = o_rho(deepset_sum_obstacle);
	memcpy(&pi_input[16], obstacles, 16 * sizeof(float));

	memcpy(&pi_input[32], goal, 4 * sizeof(float));

	const float* empty = psi(pi_input);

	// fun hack: goToGoal policy
	// const float kp = 0.5;
	// const float kv = 2.0; 
	// temp1[0] = kp*goal[0] + kv*goal[2];
	// temp1[1] = kp*goal[1] + kv*goal[3];


	// 
	float pi[2] = {empty[0],empty[1]};
	float u[2] = {0,0};

	// Scale empty to pi_max:
	float pi_norm = sqrtf(powf(pi[0], 2) + powf(pi[1], 2));
	if (pi_norm > 0) {
		const float scale = fminf(1.0f, pi_max / pi_norm);
		pi[0] = scale * pi[0];
		pi[1] = scale * pi[1];
	} 

	// Velocity
	float v[2] = {-goal[2], -goal[3]};

	// Barrier:
	float b[2] = {0, 0};
	b[0] = -barrier_kv * (v[0] + barrier_kp * barrier_grad_phi[0]) - barrier_kp * barrier_grad_phi[0] - barrier_kp * barrier_grad_phi_dot[0];
	b[1] = -barrier_kv * (v[1] + barrier_kp * barrier_grad_phi[1]) - barrier_kp * barrier_grad_phi[1] - barrier_kp * barrier_grad_phi_dot[1];

	// Complementary Filter
	// float grad_phi_norm = powf(barrier_grad_phi[0], 2) + powf(barrier_grad_phi[1], 2);
	// float vmk_norm = powf(v[0] + barrier_kp * barrier_grad_phi[0], 2) + powf(v[1] + barrier_kp * barrier_grad_phi[1], 2);
	// float a1 = barrier_kv * vmk_norm + barrier_kp * barrier_kp * grad_phi_norm;
	// float a2_1 = \
	// 	barrier_kp*(v[0]*barrier_grad_phi[0] + v[1]*barrier_grad_phi[1]);
	// float a2_2 = \
	// 	(v[0] + barrier_kp*barrier_grad_phi[0]) * (pi[0] + barrier_kp*barrier_grad_phi_dot[0]) + \
	// 	(v[1] + barrier_kp*barrier_grad_phi[1]) * (pi[1] + barrier_kp*barrier_grad_phi_dot[1]);
	// float dr = deltaR / (Rsense - robot_radius);
	// float minh = ((minp - robot_radius - deltaR) / (Rsense - robot_radius)) / dr;


	// Complementary Filter v 2 
	float grad_phi_norm = powf(barrier_grad_phi[0], 2) + powf(barrier_grad_phi[1], 2);
	float vmk_norm = powf(v[0] + barrier_kp * barrier_grad_phi[0], 2) + powf(v[1] + barrier_kp * barrier_grad_phi[1], 2);
	float a1 = barrier_kv * vmk_norm + barrier_kp * barrier_kp * grad_phi_norm;
	float a2_1 = \
		barrier_kp*(v[0]*barrier_grad_phi[0] + v[1]*barrier_grad_phi[1]);
	float a2_2 = \
		(v[0] + barrier_kp*barrier_grad_phi[0]) * (pi[0] + barrier_kp*barrier_grad_phi_dot[0]) + \
		(v[1] + barrier_kp*barrier_grad_phi[1]) * (pi[1] + barrier_kp*barrier_grad_phi_dot[1]);
	float a2 = a2_1 + a2_2;
	float minh = minp - robot_radius - deltaR;
	
	float alpha = 1. - epsilon;
	if (minh < 0. && fabsf(a2) > 0. && a1/(a1+fabs(a2)) < 1.-epsilon) {
		alpha = a1/(a1+fabs(a2));
	}

	// Composite Control Law (temp1 = pi -> temp1 = u)
	// alpha = 1.;
	// b[0] = 0.;
	// b[1] = 0.;

	u[0] = (1. - alpha) * b[0] + alpha * pi[0];
	u[1] = (1. - alpha) * b[1] + alpha * pi[1];

	// final acceleration scaling
	float u_norm = sqrtf(powf(u[0], 2) + powf(u[1], 2));
	float alpha2 = 1.0;
	if (u_norm > 0) {
		alpha2 = fminf(1.0f, max_a / u_norm);
	}
	u[0] *= alpha2;
	u[1] *= alpha2;

	// fprintf('alpha,minp,b,pi');
	// printf("alpha = %f\n",alpha);
	// printf("minp = %f\n",minp);
	// printf("b = {%f,%f}\n",b[0],b[1]);
	// printf("pi = {%f,%f}\n",pi[0],pi[1]);
	// printf("u = {%f,%f}\n",u[0],u[1]);

	memcpy(temp_dbg, u, 2 * sizeof(float));
	temp_dbg[2] = alpha;
	temp_dbg[3] = sqrtf(powf(pi[0], 2) + powf(pi[1], 2));
	temp_dbg[4] = sqrtf(powf(b[0], 2) + powf(b[1], 2));

	return temp_dbg;
}
