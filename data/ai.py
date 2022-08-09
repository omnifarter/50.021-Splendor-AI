from re import I
import time
from rules import *
import copy

DEPTH = 3  # set only to odd numbers

POINTS_VALUE = 100
T3_CARD_VALUE = 40
T2_CARD_VALUE = 30
T1_CARD_VALUE = 20  # cards must have a value higher than their tokens
GOLD_TOKEN_VALUE = 2
BASE_TOKEN_VALUE = 1

# used to give additional value to buying a card, rather than holding cards.
CAN_BUY_REGULARIZER = 15


class MinMaxBot:

    def __init__(self):
        self.board = Board()
        self.board.startGame()

    # Maximising function
    def get_moves(self, board):
        scores = []

        # ACTION.BUY_CARD
        for row in board.open_cards:
            for card in row:
                try:
                    temp_board = copy.deepcopy(board)
                    temp_board.current_player.takeAction(
                        Action.BUY_CARD, card=card)
                    action_value = self.getStateValue(temp_board)
                    scores.append({
                        'value': action_value,
                        'action': Action.BUY_CARD,
                        'kwargs': {"card": card}
                    })
                except Exception as err:
                    pass

        # ACTION.TAKE_TOKEN
        for i in range(len(board.current_player.tokens)):
            if i == 5:
                break

            # try to take 2 tokens
            try:
                token_count = [0, 0, 0, 0, 0, 0]
                token_count[i] = 2
                temp_board = copy.deepcopy(board)
                temp_board.current_player.takeAction(
                    Action.TAKE_TOKEN, tokens=token_count)
                action_value = self.getStateValue(temp_board)
                scores.append({
                    'value': action_value,
                    'action': Action.TAKE_TOKEN,
                    'kwargs': {'tokens': token_count}
                })
            except Exception as err:
                pass

            # reset board

            # try to take 1 of this token, and 2 different tokens.
            for j in range(len(board.current_player.tokens)):
                for k in range(len(board.current_player.tokens)):
                    if i == k or i == j or j == k or i == 5 or j == 5 or k == 5:
                        continue
                    try:
                        token_count = [0, 0, 0, 0, 0, 0]
                        token_count[i] = 1
                        token_count[j] = 1
                        token_count[k] = 1
                        temp_board = copy.deepcopy(board)
                        temp_board.current_player.takeAction(
                            Action.TAKE_TOKEN, tokens=token_count)
                        action_value = self.getStateValue(temp_board)
                        scores.append({
                            'value': action_value,
                            'action': Action.TAKE_TOKEN,
                            'kwargs': {'tokens': token_count}
                        })
                    except Exception as err:
                        pass
                    # reset board

        return scores

    def minimax(self, depth, board, maximize, parent_score):
        current_actions = self.get_moves(board)

        if depth == DEPTH and maximize:
            return max(current_actions, key=lambda score: score['value']) if len(current_actions) != 0 else {"value": -1e10}

        if depth == DEPTH and not maximize:
            return min(current_actions, key=lambda score: score['value']) if len(current_actions) != 0 else {"value": 1e10}

        best_opponent_action = {"value": -
                                1e10} if maximize else {"value": 1e10}
        best_current_action = {"value": -1e10} if maximize else {"value": 1e10}

        for move in current_actions:
            temp_board = copy.deepcopy(board)

            temp_board.current_player.takeAction(
                move['action'], **move['kwargs'])

            try:
                opponent_action = self.minimax(
                    depth+1, temp_board, not maximize, best_current_action)
            except Exception as err:
                print("ERROR 111: ", err, move)
                raise err
            if maximize:
                if opponent_action['value'] > best_opponent_action['value']:
                    best_opponent_action = copy.deepcopy(opponent_action)
                    best_current_action = copy.deepcopy(move)
            else:
                if opponent_action['value'] < best_opponent_action['value']:
                    best_opponent_action = copy.deepcopy(opponent_action)
                    best_current_action = copy.deepcopy(move)

            # alpha-beta pruning

            # if the parent has already set one of its value
            if parent_score['value'] != -1e10 and parent_score['value'] != 1e10:

                if maximize:
                    if parent_score['value'] < best_current_action['value']:
                        print("PRUNED")
                        break
                else:
                    if parent_score['value'] > best_current_action['value']:
                        print("PRUNED")
                        break

        return best_current_action

    def ai_move(self):
        if self.board.current_player.id != 1:
            raise Exception('NOT_AI_TURN')
        start = time.time()
        best_move = self.minimax(1, self.board, True, {"value": -1e10})

        print('calculated best move in {} s', time.time() - start)

        print('best_move', best_move)
        self.board.current_player.takeAction(
            best_move['action'], **best_move['kwargs'])

    # Gets the next board state value
    def getStateValue(self, board: Board):

        # human: player1
        total_token_value = sum(
            [BASE_TOKEN_VALUE * t if i != 5 else GOLD_TOKEN_VALUE * t for i, t in enumerate(board.player1.tokens)])

        total_card_value = 0
        for card in board.player1.cards:
            if card.tier == 1:
                total_card_value += T1_CARD_VALUE
            elif card.tier == 2:
                total_card_value += T2_CARD_VALUE
            else:
                total_card_value += T3_CARD_VALUE

        can_buy_value = 0
        can_buy_flag = False
        for row in board.open_cards:
            for card in row:
                try:
                    # if player can take a card next turn, we assign a arbitary value
                    spent_tokens = board.player1.tokensForCard(card)
                    can_buy_value -= (sum(spent_tokens) + CAN_BUY_REGULARIZER)
                    if card.tier == 1:
                        can_buy_value += T1_CARD_VALUE
                    elif card.tier == 2:
                        can_buy_value += T2_CARD_VALUE
                    else:
                        can_buy_value += T3_CARD_VALUE
                    can_buy_flag = True
                    break
                except:
                    continue
            if can_buy_flag:
                break

        player1_points = POINTS_VALUE * board.player1.points + \
            total_card_value + total_token_value + can_buy_value

        # AI: player2
        total_token_value = sum(
            [BASE_TOKEN_VALUE * t if i != 5 else GOLD_TOKEN_VALUE * t for i, t in enumerate(board.player2.tokens)])

        total_card_value = 0
        for card in board.player2.cards:
            if card.tier == 1:
                total_card_value += T1_CARD_VALUE
            elif card.tier == 2:
                total_card_value += T2_CARD_VALUE
            else:
                total_card_value += T3_CARD_VALUE

        can_buy_value = 0
        can_buy_flag = False
        for row in board.open_cards:
            for card in row:
                try:
                    # if player can take a card next turn, we assign a arbitary value
                    spent_tokens = board.player2.tokensForCard(card)
                    can_buy_value -= (sum(spent_tokens) + CAN_BUY_REGULARIZER)
                    if card.tier == 1:
                        can_buy_value += T1_CARD_VALUE
                    elif card.tier == 2:
                        can_buy_value += T2_CARD_VALUE
                    else:
                        can_buy_value += T3_CARD_VALUE
                    # We must break here, else the AI will keep collecting
                    # as much "can_buy" cards without actually buying them.
                    can_buy_flag = True
                    break

                except:
                    continue
            if can_buy_flag:
                break

        player2_points = POINTS_VALUE * board.player2.points + \
            total_card_value + total_token_value + can_buy_value

        return player2_points - player1_points


ai = MinMaxBot()
