#include <cmath>
#include <cstddef>
#include <vector>

#include <fstream>
#include <random>

#include "RVO.h"

/* Store the goals of the agents. */
std::vector<RVO::Vector2> goals;


static const int numAgents = 10;
static const float radius = 15;

void setupScenario(RVO::RVOSimulator *sim)
{
  /* Specify the global time step of the simulation. */
  sim->setTimeStep(0.25f);

  /* Specify the default parameters for agents that are subsequently added. */
  sim->setAgentDefaults(
    /* neighborDist*/ 15.0f,
    /* maxNeighbors*/ 10,
    /* timeHorizon*/ 10.0f,
    /* timeHorizonObst*/ 10.0f,
    /* radius*/ 1.5f,
    /* maxSpeed*/ 2.0f);

  // /*
  //  * Add agents, specifying their start position, and store their goals on the
  //  * opposite side of the environment.
  //  */
  // for (size_t i = 0; i < numAgents; ++i) {
  //   sim->addAgent(radius *
  //                 RVO::Vector2(std::cos(i * 2.0f * M_PI / numAgents),
  //                              std::sin(i * 2.0f * M_PI / numAgents)));
  //   goals.push_back(-sim->getAgentPosition(i));
  // }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-radius, radius);
  for (size_t i = 0; i < numAgents;) {
    float x = dis(gen);
    float y = dis(gen);
    RVO::Vector2 pos(x, y);
    bool collision = false;
    for (size_t j = 0; j < i; ++j) {
      float dist = RVO::abs(pos - sim->getAgentPosition(j));
      if (dist <= 2) {
        collision = true;
        break;
      }
    }
    if (!collision) {
      sim->addAgent(pos);
      // find a collision-free goal
      do {
        RVO::Vector2 goal(dis(gen), dis(gen));
        collision = false;
        for (size_t j = 0; j < i; ++j) {
          float dist = RVO::abs(goal - goals[j]);
          if (dist <= 2) {
            collision = true;
            break;
          }
        }
        if (!collision) {
          goals.push_back(goal);
          break;
        }
      } while(true);
      // next agent
      ++i;
    }
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

    /*
     * Perturb a little to avoid deadlocks due to perfect symmetry.
     */
    float angle = std::rand() * 2.0f * M_PI / RAND_MAX;
    float dist = std::rand() * 0.1f / RAND_MAX;

    sim->setAgentPrefVelocity(i, sim->getAgentPrefVelocity(i) +
                              dist * RVO::Vector2(std::cos(angle), std::sin(angle)));
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

  // keep simulation running a bit longer
  for (size_t i = 0; i < 10; ++i) {
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

  return 0;
}
