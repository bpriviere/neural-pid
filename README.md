# Neural-PID

Project that explores learning a motion planner and PID controller with time-varying gains from an existing policy (such as one learned by RL).

## Dependencies

* Python3
* PyTorch: https://pytorch.org/
* OpenAI Gym: https://gym.openai.com/docs/
* MeshCat (optional, for animations): https://github.com/rdeits/meshcat-python

```
sudo pip3 install torch==1.2.0+cpu torchvision==0.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
sudo pip3 install gym
sudo pip3 install meshcat
```

## Execute

### Train policy using RL

```
python3 examples/motionplanner.py --rl
```

### Train PID controller an motion planner

```
python3 examples/motionplanner.py --il
```

### Evaluate RL and PID solutions

```
python3 examples/motionplanner.py [--animate]
```
