#ifndef __NN_H__
#define __NN_H__


void nn_reset(void);

void nn_add_neighbor(const float input[4]);

void nn_add_obstacle(const float input[4]);

const float* nn_eval(const float input[4]);

#endif