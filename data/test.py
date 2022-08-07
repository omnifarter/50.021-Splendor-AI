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
print('Player 0 taking 2 White tokens')
board.current_player.takeAction(Action.TAKE_TOKEN,tokens=[0,2,0,0,0,0])
print(board.player1)

print('Player 1 taking Blue, Black, Red tokens')
board.current_player.takeAction(Action.TAKE_TOKEN,tokens=[0,0,1,1,1,0])
print(board.player2)

"""
Testing buying of cards
"""
board.current_player.takeAction(Action.TAKE_TOKEN,tokens=[1,0,1,1,0,0])
board.current_player.takeAction(Action.TAKE_TOKEN,tokens=[1,1,0,0,1,0])
print('Player 0 has enough to buy card: ',board.player1)
board.current_player.takeAction(Action.BUY_CARD,card=board.open_cards[0][0])
print('Player 0 has bought card:', board.player1)