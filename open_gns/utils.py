
import ipyvolume as ipv
import numpy as np

def animate_rollout(rollout):
    """Takes rollout positions and renders a 3d animation"""
    x, y, z = [], [], []
    for i, step in enumerate(rollout):
        pos = step.numpy()
        x.append(pos[:,0])
        y.append(pos[:,1])
        z.append(pos[:,2])

    fig = ipv.figure()
    color = [(1,0,0) if i < 64 else (0,0,1) for i in range(len(pos))]
    s = ipv.scatter(np.array(x), np.array(y), np.array(z), color=color, size=5, marker='sphere') 
    ipv.animation_control(s)
    ipv.show()
    
def animate_rollout_quiver(positions, vector):
    """Takes positions & vector(acceleration or velocity) and renders 3d quiver animation"""
    num_steps = len(positions)
    num_particles = positions[0].size(0)
    data = np.zeros((6, num_steps, num_particles))
    for s in range(num_steps):
        pos = positions[s].numpy()
        v = vector[s].numpy()
        for i in range(3):
            data[i][s] = pos[:,i]
            data[i+3][s] = v[:, i]
    fig = ipv.figure()
    color = [(1,0,0) if i < 64 else (0,0,1) for i in range(len(pos))]
    s = ipv.quiver(*data, color=color, size=5) 
    ipv.animation_control(s)
    ipv.show()
    