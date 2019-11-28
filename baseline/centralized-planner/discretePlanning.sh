# discrete planning
./multi-robot-trajectory-planning/build/libMultiRobotPlanning/ecbs -i $1 -o multi-robot-trajectory-planning/examples/ground/output/discreteSchedule.yaml -w 1.1
# python3 multi-robot-trajectory-planning/libMultiRobotPlanning/example/visualize.py $1 multi-robot-trajectory-planning/examples/ground/output/discreteSchedule.yaml
# convert yaml map -> octomap
./multi-robot-trajectory-planning/build/tools/map2octomap/map2octomap -m $1 -o multi-robot-trajectory-planning/examples/ground/map.bt
# convert octomap -> STL (for visualization)
./multi-robot-trajectory-planning/build/tools/octomap2openscad/octomap2openscad -i multi-robot-trajectory-planning/examples/ground/map.bt -o multi-robot-trajectory-planning/examples/ground/output/map.scad
openscad -o multi-robot-trajectory-planning/examples/ground/output/map-ascii.stl multi-robot-trajectory-planning/examples/ground/output/map.scad
stl2bin multi-robot-trajectory-planning/examples/ground/output/map-ascii.stl multi-robot-trajectory-planning/examples/ground/output/map.stl