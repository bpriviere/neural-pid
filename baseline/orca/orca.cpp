#include <cmath>
#include <cstddef>
#include <vector>

#include <fstream>

#include "RVO.h"

/* Store the goals of the agents. */
std::vector<RVO::Vector2> goals;

void setupScenario(RVO::RVOSimulator *sim)
{
  /* Specify the global time step of the simulation. */
  sim->setTimeStep(0.25f);

  /* Specify the default parameters for agents that are subsequently added. */
  sim->setAgentDefaults(15.0f, 10, 10.0f, 10.0f, 1.5f, 2.0f);

  /*
   * Add agents, specifying their start position, and store their goals on the
   * opposite side of the environment.
   */
  for (size_t i = 0; i < 250; ++i) {
    sim->addAgent(200.0f *
                  RVO::Vector2(std::cos(i * 2.0f * M_PI / 250.0f),
                               std::sin(i * 2.0f * M_PI / 250.0f)));
    goals.push_back(-sim->getAgentPosition(i));
  }
}

void setPreferredVelocities(RVO::RVOSimulator *sim)
{
  /*
   * Set the preferred velocity to be a vector of unit magnitude (speed) in the
   * direction of the goal.
   */
  for (int i = 0; i < static_cast<int>(sim->getNumAgents()); ++i) {
    RVO::Vector2 goalVector = goals[i] - sim->getAgentPosition(i);

    if (RVO::absSq(goalVector) > 1.0f) {
      goalVector = RVO::normalize(goalVector);
    }

    sim->setAgentPrefVelocity(i, goalVector);
  }
}

bool reachedGoal(RVO::RVOSimulator *sim)
{
  /* Check if all agents have reached their goals. */
  for (size_t i = 0; i < sim->getNumAgents(); ++i) {
    if (RVO::absSq(sim->getAgentPosition(i) - goals[i]) > sim->getAgentRadius(i) * sim->getAgentRadius(i)) {
      return false;
    }
  }

  return true;
}

int main()
{
  /* Create a new simulator instance. */
  RVO::RVOSimulator sim;

  /* Set up the scenario. */
  setupScenario(&sim);

  std::ofstream output("orca.csv");
  output << "t";
  for (size_t i = 0; i < sim.getNumAgents(); ++i) {
    output << ",x" << i << ",y" << i << ",vx" << i << ",vy" << i;
  }
  output << std::endl;

  /* Perform (and manipulate) the simulation. */
  do {
    // output current simulation result
    output << sim.getGlobalTime();
    for (size_t i = 0; i < sim.getNumAgents(); ++i) {
      auto pos = sim.getAgentPosition(i);
      auto vel = sim.getAgentVelocity(i);
      output << "," << pos.x() << "," << pos.y() << "," << vel.x() << "," << vel.y();
    }
    output << std::endl;

    setPreferredVelocities(&sim);
    sim.doStep();
  }
  while (!reachedGoal(&sim));

  return 0;
}