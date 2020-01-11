#ifndef __NN1_H__
#define __NN1_H__


void nn_reset(void);

void nn_add_neighbor(const float input[]);

void nn_add_obstacle(const float input[]);

const float* nn_eval(const float* input);

#endif