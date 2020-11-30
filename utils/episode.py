import replay

# play an episode

def play_ep(env, policy, render = False):
    states, actions, rewards = [], [], []
    states.append(env.reset())
    done = False
    while not done:
        action = policy(env, states[-1])
        actions.append(action)
        if render: env.render()
        obs, reward, done, info = env.step(action)
        states.append(obs)
        rewards.append(reward)
    return states, actions, rewards

# play an episode + add to memory

def play_ep_mem(env, policy, memory):
    states, actions, rewards = [], [], []
    states.append(env.reset())
    done = False
    while not done:
        action = policy(env, states[-1])
        actions.append(action)
        obs, reward, done, info = env.step(action)
        memory.add(states[-1], action, reward, obs, done)
        states.append(obs)
        rewards.append(reward)
    return states, actions, rewards