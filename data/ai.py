from rules import *
import copy

POINTS_VALUE     = 10
T3_CARD_VALUE    = 5
T2_CARD_VALUE    = 4
T1_CARD_VALUE    = 3
GOLD_TOKEN_VALUE = 2
BASE_TOKEN_VALUE = 1

class MinMaxBot:

    def __init__(self):
        self.board = Board()
        self.board.startGame()

    # Maximising function
    def max(self):
        value = -1
        action = -1
        kwargs = {}
        
        temp_board = copy.deepcopy(self.board)

        # ACTION.BUY_CARD
        for row in temp_board.open_cards:
            for card in row:
                try:
                    temp_board.current_player.takeAction(Action.BUY_CARD, card=card)
                    action_value = self.getStateValue(temp_board.player1)
                    if action_value > value:
                        value = action_value
                        action = Action.BUY_CARD
                        kwargs['card'] = card
                except:
                    pass

                temp_board = copy.deepcopy(self.board)

        # ACTION.TAKE_TOKEN
        for i in len(temp_board.current_player.tokens):
            if i == 5:
                break
            
            # try to take 2 tokens
            try:
                token_count = [0,0,0,0,0,0]
                token_count[i] = 2
                temp_board.current_player.takeAction(Action.TAKE_TOKEN, token_count)
                action_value = self.getStateValue(temp_board.player1)
                if action_value > value:
                    value = action_value
                    action = Action.TAKE_TOKEN
                    kwargs['tokens'] = token_count
            except:
                pass

            # reset board
            temp_board = copy.deepcopy(self.board)
     
            # try to take 1 of this token, and 2 different tokens.
            for j in len(temp_board.current_player.tokens):
                for k in len(temp_board.current_player.tokens):
                    try:
                        token_count = [0,0,0,0,0,0]
                        token_count[i] = 1
                        token_count[j] = 1
                        token_count[k] = 1
                        temp_board.current_player.takeAction(Action.TAKE_TOKEN, token_count)
                        action_value = self.getStateValue(temp_board.player1)
                        if action_value > value:
                            value = action_value
                            action = Action.TAKE_TOKEN
                            kwargs['tokens'] = token_count
                    except:
                        pass
                    # reset board
                    temp_board = copy.deepcopy(self.board)

        return (value,action,kwargs)

    def min(self):
        value = 1e10
        action = -1
        kwargs = {}

        #TODO: ACTION.BUY_CARD
        #TODO: ACTION.TAKE_TOKEN

        return (value,action,kwargs)

    # Gets the next board state value
    def getStateValue(self,player: PlayerState):
        total_token_value = sum([BASE_TOKEN_VALUE * t if i != 5 else GOLD_TOKEN_VALUE * t for i, t in enumerate(player.tokens)]) 

        total_card_value = T1_CARD_VALUE * sum(player.cards[0]) 
        + T2_CARD_VALUE * sum(player.cards[1]) 
        + T3_CARD_VALUE * sum(player.cards[2])
        
        return (
            POINTS_VALUE * player.points +
            total_card_value +
            total_token_value 
        ) 