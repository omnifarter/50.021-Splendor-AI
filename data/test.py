from rules import *

board = Board()

board.startGame()

"""
Test board state
"""
print('Opened Cards:')
print(board.open_cards)

"""
Testing Tokens
"""
print('Player 0 taking 2 Green tokens')
board.currentPlayer.takeAction(Action.TAKE_TOKEN,tokens=[2,0,0,0,0,0])
print(board.player1)

print('Player 1 taking Blue, Black, Red tokens')
board.currentPlayer.takeAction(Action.TAKE_TOKEN,tokens=[0,0,1,1,1,0])
print(board.player2)
try:
    board.currentPlayer.takeAction(Action.TAKE_TOKEN,tokens=[0,0,1,1,1,0])
except:
    print('test passed - invalid move')
    pass

"""

"""