import matplotlib.pyplot as plt
import pickle

def plot_traj(dist, time):
    plt.plot(time, dist)
    plt.xlabel("Time (s)")
    plt.ylabel("Distance (m)")
    plt.title("distance to goal over time")
    plt.savefig("./traj.png")

with open("./traj2.pkl", "rb") as f:
    data = pickle.load(f)

dist, time = data
plot_traj(dist, time)