# create examples for global planner
# python3 animate.py --video central_obst6_agents4.mp4 --speed 8 ../../data/singleintegrator/instances3/map_8by8_obst06_agents004_ex000000.yaml ../../data/singleintegrator/central3/map_8by8_obst06_agents004_ex000000.npy
# python3 animate.py --video central_obst6_agents8.mp4 --speed 8 ../../data/singleintegrator/instances3/map_8by8_obst06_agents008_ex000001.yaml ../../data/singleintegrator/central3/map_8by8_obst06_agents008_ex000001.npy
# python3 animate.py --video central_obst6_agents16.mp4 --speed 8 ../../data/singleintegrator/instances3/map_8by8_obst06_agents016_ex000000.yaml ../../data/singleintegrator/central3/map_8by8_obst06_agents016_ex000000.npy
# python3 animate.py --video central_obst12_agents4.mp4 --speed 8 ../../data/singleintegrator/instances3/map_8by8_obst12_agents004_ex000000.yaml ../../data/singleintegrator/central3/map_8by8_obst12_agents004_ex000000.npy
# python3 animate.py --video central_obst12_agents8.mp4 --speed 8 ../../data/singleintegrator/instances3/map_8by8_obst12_agents008_ex000001.yaml ../../data/singleintegrator/central3/map_8by8_obst12_agents008_ex000001.npy
# python3 animate.py --video central_obst12_agents16.mp4 --speed 8 ../../data/singleintegrator/instances3/map_8by8_obst12_agents016_ex000000.yaml ../../data/singleintegrator/central3/map_8by8_obst12_agents016_ex000000.npy

# # create videos for observations
# python3 createPlots.py ../../data/singleintegrator/instances3/map_8by8_obst06_agents004_ex000000.yaml ../../data/singleintegrator/central3/map_8by8_obst06_agents004_ex000000.npy
# ffmpeg -y -r 10 -i obs_central_f%d.png obs_central.mp4
# ffmpeg -y -r 10 -i obs_agent0_f%d.png obs_agent0.mp4
# ffmpeg -y -r 10 -i obs_agent1_f%d.png obs_agent1.mp4
# ffmpeg -y -r 10 -i obs_agent2_f%d.png obs_agent2.mp4
# ffmpeg -y -r 10 -i obs_agent3_f%d.png obs_agent3.mp4
# rm obs_central*.png
# rm obs_agent*.png

python3 animate.py --video large.mp4 --speed 8 ../../results/singleintegrator/instances/map_35by20_obst70_agents350_ex000000.yaml ../../results/singleintegrator/cBF/map_35by20_obst70_agents350_ex000000.npy
