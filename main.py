from cartpole import CartPoleEnvironment as game
from DQN_agent import DQN_agent, DQN_trainer

agent = DQN_agent(input_space=4, action_space=3)
trainer = DQN_trainer(game(), agent=agent, sample_size=5000)
trainer.epsilon_scheduler.reset()

trainer.train(epochs=10000, ep_len=500)
#trainer.test(ep_len=500, display=True)

