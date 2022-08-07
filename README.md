# 50.021-Splendor-AI

## Project Description
Splendor is a board game of token-collecting and card development, which can be played with 2-4 players. In this project, we attempt to create an AI that is able to understand the goal of the game and play substantially well against humans. 

The original game rules include many different actions each player can take during their turn. However, for the scope of this project, we have simplified the rules to form a relaxed problem of the board game Splendor. 

For this project, the game will be played by 2 players. At each turn, a player can choose to draw tokens from the bank, or use their tokens to buy a card. There are 12 total open cards on the board, which the player can buy if they have the correct tokens. The cards are laid out in rows (see fig. 1), with each row corresponding to a different tier. Each card has a point value from 0 to 5. The first player to reach 15 points wins.


## Attempted Solutions
- MinMax Algorithm (has GUI)
    - see `./data` and `./ui`
- Monte-Carlo Tree Search
    - see: `./model/mcts.py` and `./model/mcts_rules.py`
- Reinforcement Learning: Deep Q-Learning
    - see: `./model/SplendorDQL.py` and `./model/rl_rules.py`
    - saved outputs at: `./save`

## Running the GUI with MinMax Algorithm

Navigate to the `./data` folder and run `flask run` to start the backend.

In a separate terminal, navigate to the `./ui` folder and run:
```
npm i
npm start
```


## References
- https://maxcandocia.com/article/2018/May/04/reinforcement-learning-for-splendor/
- https://github.com/filipmlynarski/splendor-ai