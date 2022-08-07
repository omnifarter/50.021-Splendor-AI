from SplendorDQL import SplendorDQN

episodes = 300
dqn = SplendorDQN(num_episodes=episodes, name=f'SingleDQN_{episodes}games')
dqn.train()