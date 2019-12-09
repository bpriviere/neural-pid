#!/usr/bin/env python3
import yaml
import argparse

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("input", help="input file containing schedule")
  parser.add_argument("output", help="output file with post-processed schedule")
  args = parser.parse_args()


  with open(args.input) as input_file:
    data = yaml.load(input_file, Loader=yaml.SafeLoader)

  for agent_name in data["schedule"]:
    agent = data["schedule"][agent_name]
    path = []
    for i in range(0, len(agent)-1):
      path.append({
        'x': agent[i]['x'],
        'y': agent[i]['y'],
        't': i*2})
      path.append({
        'x': (agent[i]['x'] + agent[i+1]['x'])/2,
        'y': (agent[i]['y'] + agent[i+1]['y'])/2,
        't': i*2+1})
    path.append({
      'x': agent[-1]['x'],
      'y': agent[-1]['y'],
      't': (i+1)*2})
    data["schedule"][agent_name] = path

  with open(args.output, 'w') as output_file:
    yaml.dump(data, output_file)
