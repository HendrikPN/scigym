import scigym
KWARGS = {'board_size': 3, 'error_model': 1, 'error_rate': 0.001}
env = scigym.make("toricgame-v0", **KWARGS)

# Initialize a game that is not already solved
observation = env.reset()
    
# Render the initial state
env.render()

done = False
while not done:
    action = int(input(f"Pick action from {env.action_space.n} actions: "))
    observation, reward, done, info = env.step(action)
    print(reward)
    env.render()

print("Combined with the original physical qubit errors, this is the total error string:")
for q in env.initial_qubits_flips[0]:
    env.state.act(q, 0, update_syndrome=False)
env.render()
    
if reward == 1:
    print("You win!")
else: 
    print("You've lost")