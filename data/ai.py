from rules import *
import copy

DEPTH = 3  # set only to odd numbers

POINTS_VALUE = 100
T3_CARD_VALUE = 30
T2_CARD_VALUE = 20
T1_CARD_VALUE = 10  # cards must have a value higher than their tokens
GOLD_TOKEN_VALUE = 2
BASE_TOKEN_VALUE = 1


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
            if opponent_action['action'] == Action.BUY_CARD:
                print('temp board player 1', temp_board.player1.serialize())
                print('temp board player 2', temp_board.player2.serialize())
                print('temp board turn', temp_board.current_player.serialize())
                if depth == 1:
                    print(
                        "I am playing as an AI. i want the human to have as low of a score as possible.")
                    print('This is the move that I just made', move)
                    print(
                        "Of all possible moves, this is the best current move so far", best_current_action)
                    print(
                        "if the move i just made results in a lower value move for the opponent, i will do it")
                    print("This is opponent's action", opponent_action)
                    print("This is best opponent action", best_opponent_action)

                if depth == 2:
                    print(
                        "I am playing as a human. i want the AI to have as low of a score as possible.")
                    print('This is the move that I just made', move)
                    print(
                        "Of all possible moves, this is the best current move so far", best_current_action)
                    print(
                        "if the move i just made results in a lower value move for the opponent, i will do it")
                    print("This is opponent's action", opponent_action)
                    print("This is best opponent action", best_opponent_action)
            if maximize:
                if opponent_action['value'] > best_opponent_action['value']:
                    best_opponent_action = copy.deepcopy(opponent_action)
                    best_current_action = copy.deepcopy(move)
                    if depth == 1:
                        print(
                            "I am playing as an AI. i want the human to have as low of a score as possible.")
                        print('This is the move that I just made', move)
                        print(
                            "Of all possible moves, this is the best current move so far", best_current_action)
                        print(
                            "if the move i just made results in a lower value move for the opponent, i will do it")
                        print("This is opponent's action", opponent_action)
                        print("This is best opponent action", best_opponent_action)


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
        # if depth == 2:
        #     print('as the human, i deemed that this the opponent will make this move',
        #           best_opponent_action)
        #     print('Thus, i will make this move', best_current_action)
        # if depth == 1:
        #     print('as the AI, i deemed that this the opponent will make this move',
        #           best_opponent_action)
        #     print('Thus, i will make this move', best_current_action)

        return best_current_action

    def ai_move(self):
        if self.board.current_player.id != 1:
            raise Exception('NOT_AI_TURN')
        best_move = self.minimax(1, self.board, True, {"value": -1e10})
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

        player1_points = POINTS_VALUE * board.player1.points + \
            total_card_value + total_token_value

        #TODO: I need to assign extra value to the state, if the player is able to buy a card the next turn.
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

        player2_points = POINTS_VALUE * board.player2.points + \
            total_card_value + total_token_value

        return player2_points - player1_points


ai = MinMaxBot()
