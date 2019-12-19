#!/usr/bin/env python3
import yaml
import argparse

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("input", help="input file containing schedule")
  parser.add_argument("output", help="output file with pre-processed schedule")
  args = parser.parse_args()

  with open(args.input) as input_file:
    data = yaml.load(input_file, Loader=yaml.SafeLoader)

  for agent in data["agents"]:
    agent['start_real'] = agent['start']
    agent['goal_real'] = agent['goal']
    agent['start'] = [round(v) for v in agent['start']]
    agent['goal'] = [round(v) for v in agent['goal']]

  with open(args.output, 'w') as output_file:
    yaml.dump(data, output_file)
