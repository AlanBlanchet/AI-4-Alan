import gymnasium

name = "ALE/Breakout-v5"

env = gymnasium.make(name, render_mode="rgb_array")


print(env)
