#include <cmath>
#include <vector>

#include "AVO.h"

#include <fstream>

const float AVO_TWO_PI = 6.283185307179586f;

bool haveReachedGoals(const AVO::Simulator &simulator,
                      const std::vector<AVO::Vector2> &goals) {
  for (std::size_t i = 0; i < simulator.getNumAgents(); ++i) {
    if (AVO::absSq(simulator.getAgentPosition(i) - goals[i]) > 0.25f) {
      return false;
    }
  }

  return true;
}

int main() {
  AVO::Simulator sim;

  sim.setTimeStep(0.25f);

  sim.setAgentDefaults(
    /* neighborDist*/ 15.0f,
    /* maxNeighbors*/ 10,
    /* timeHorizon*/ 10.0f,
    /* radius*/ 1.5f,
    /* maxSpeed*/ 2.0f,
    /* maxAccel*/ 1.0f,
    /* accelInterval*/ 0.25f);

  std::vector<AVO::Vector2> goals;

  const int numAgents = 4;
  const float radius = 10;

  for (std::size_t i = 0; i < numAgents; ++i) {
    const AVO::Vector2 position =
        radius * AVO::Vector2(std::cos(i * AVO_TWO_PI / numAgents),
                              std::sin(i * AVO_TWO_PI / numAgents));

    sim.addAgent(position);
    goals.push_back(-position);
  }

  std::ofstream output("avo.csv");
  output << "t";
  for (size_t i = 0; i < sim.getNumAgents(); ++i) {
    output << ",x" << i << ",y" << i << ",vx" << i << ",vy" << i;
  }
  output << std::endl;

  int stayingAtGoal = 0;
  do {
    // output current simulation result
    output << sim.getGlobalTime();
    for (size_t i = 0; i < sim.getNumAgents(); ++i) {
      auto pos = sim.getAgentPosition(i);
      auto vel = sim.getAgentVelocity(i);
      output << "," << pos.getX() << "," << pos.getY() << "," << vel.getX() << "," << vel.getY();
    }
    output << std::endl;

    // set preferred velocities
    for (std::size_t i = 0; i < sim.getNumAgents(); ++i) {
      AVO::Vector2 toGoal = goals[i] - sim.getAgentPosition(i);

      if (AVO::absSq(toGoal) > 1.0f) {
        toGoal = normalize(toGoal);
      }

      sim.setAgentPrefVelocity(i, toGoal);

      // Perturb a little to avoid deadlocks due to perfect symmetry.
      float angle = std::rand() * 2.0f * M_PI / RAND_MAX;
      float dist = std::rand() * 0.1f / RAND_MAX;

      sim.setAgentPrefVelocity(i, sim.getAgentPrefVelocity(i) +
                              dist * AVO::Vector2(std::cos(angle), std::sin(angle)));
    }

    sim.doStep();

    if (haveReachedGoals(sim, goals)) {
      stayingAtGoal++;
    } else {
      stayingAtGoal = 0;
    }
  } while (stayingAtGoal < 20);

  return 0;
}
