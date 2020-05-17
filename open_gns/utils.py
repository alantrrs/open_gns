
import ipyvolume as ipv
import numpy as np

def animate_rollout(rollout):
    """Takes rollout positions and renders a 3d animation"""
    x, y, z = [], [], []
    for step in rollout:
        pos = step.numpy()
        x.append(pos[:,0])
        y.append(pos[:,1])
        z.append(pos[:,2])

    fig = ipv.figure()
    color = [(1,0,0) if i < 64 else (0,0,1) for i in range(len(pos))]
    s = ipv.scatter(np.array(x), np.array(y), np.array(z), color=color, size=5, marker='sphere') 
    ipv.animation_control(s)
    ipv.show()
    