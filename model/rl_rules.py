import itertools
import numpy as np
import random

card_path = '../data/cards.csv'
nobles_path = '../data/nobles.csv'


class Card:
    def __init__(self, id, data):
        # id: card ID, autogenerated
        # cost: array of ints representing the token cost of the card

        self.id = id
        self.tier = data[0]
        self.value = data[1]
        self.type = data[2] - 1
        self.cost = data[3:]

    def __repr__(self):
        return f"Card {self.id}:\nTier: {self.tier} \nValue: {self.value}\nType: {self.type}\nCost: {self.cost}"

    def serialize(self):
        return [self.value, self.type, *self.cost]


class Noble:
    def __init__(self, id, cost):
        # cost: array of ints representing total card cost for each type required to buy the noble
        self.id = id
        self.cost = cost
        self.points = 3

    def __str__(self):
        return f"ID: {self.id}, Cost: {self.cost}"

    def serialize(self):
        return [*self.cost, self.points]


class TokenBank:
    def __init__(self):
        self.tokens = [4] * 5 + [5]

    def serialize(self):
        return self.tokens

    # update the tokens in the bank.
    def update(self, token_change, subtract=False):
        if subtract:
            token_change = [-x for x in token_change]
        self.tokens = [sum(x) for x in zip(self.tokens, token_change)]


class PlayerState:
    def __init__(self, id, turn_order):
        self.id = id
        self.turn_order = turn_order
        self.points = 0

        # Player inventory
        self.cards = []
        self.card_counts = [0, 0, 0, 0, 0]
        self.reserved_cards = []
        self.tokens = [0, 0, 0, 0, 0, 0]
        self.nobles = []

    def __str__(self):
        return "\nPlayer {}:\nPoints: {}\nTokens: {}\nCards: {}Reserves: {}".format(self.id, self.points, self.tokens,
                                                                                    self.cards, self.reserved_cards)

    def serialize(self):
        return [
            *self.card_counts,
            self.points,
            *self.tokens,
        ]

    def updateNoble(self, noble: Noble):
        # used for updating points if player gets a noble 
        self.points += 3
        self.nobles.append(noble)

    def updateTokens(self, token_change, subtract=False):
        if subtract:
            token_change = [-x for x in token_change]
        self.tokens = [sum(x) for x in zip(self.tokens, token_change)]

    def updateCards(self, card: Card):
        self.cards.append(card)
        self.card_counts[card.type] += 1
        self.points += card.value


# Master class that controls the game
# 1. Control changes to board state
# 2. Controls changes to player state
class Board:
    def __init__(self):
        self.all_cards, self.all_nobles = self._read_data()
        self.open_cards = [[], [], []]
        self.deck_cards = [[], [], []]
        self.nobles = []

        self.bank = TokenBank()
        self.player1 = PlayerState(id=0, turn_order=0)
        self.player2 = PlayerState(id=1, turn_order=1)
        self.players = itertools.cycle([self.player1, self.player2])
        self.current_player = self.players.__next__()
        self.turn = 1
        self.points_to_win = 15

        # Init board state
        # split cards into tier decks
        for card in self.all_cards:
            self.deck_cards[card.tier - 1].append(card)

        # draw 4 cards per tier, add to open_cards
        for tier in range(3):
            self.open_cards[tier].extend(self._draw_cards(tier, 4))

        # choose 3 nobles
        for i in range(3):
            idx = random.randint(0, len(self.all_nobles))
            self.nobles.append(self.all_nobles.pop(idx))

    def getState(self):
        # returns 1d list of board states
        # 12 open cards (12, 7), 3 nobles (3, 6), player1 (1, 12), player2 (1, 12), bank (1, 6)
        dims = (12 * 7) + 2 * 12 + 6 + 18
        data = np.zeros(dims)

        idx = 0
        for tier in self.open_cards:
            for card in tier:
                data[idx: idx + 7] = card.serialize()
                idx += 7

        for noble in self.nobles:
            data[idx: idx + 6] = noble.serialize()
            idx += 6

        data[idx: idx + 12] = self.player1.serialize()
        idx += 12
        data[idx: idx + 12] = self.player2.serialize()
        idx += 12
        data[idx:] = self.bank.serialize()

        return data

    def playerAction(self, action_index):
        # TODO
        if action_index == 1:
            # take tier 1 card 0
            card = self.open_cards[0].pop(0)

            # Update player states
            self.current_player.updateCards(card)
            self.current_player.updateTokens(card.cost, subtract=True)

            # Update board states
            self._draw_cards(1, 1)
            self.bank.update(card.cost)

        elif action_index == 2:
            pass

        # at the end of player action, check nobles
        self._check_nobles()
        self._end_player()

    def _read_data(self):
        temp_cards = np.genfromtxt(card_path, dtype=np.int32, delimiter=',', skip_header=1).tolist()
        temp_nobles = np.genfromtxt(nobles_path, dtype=np.int32, delimiter=',', skip_header=1).tolist()

        cards = []
        for i in range(len(temp_cards)):
            c = Card(i, temp_cards[i])
            cards.append(c)

        nobles = []
        for i in range(len(temp_nobles)):
            n = Noble(i, temp_nobles[i])
            nobles.append(n)

        return cards, nobles

    def _draw_cards(self, tier, num_cards):
        # remove cards from the specified tier deck and add to list of open cards
        idx = []
        cards = []
        for i in range(num_cards):
            idx.append(random.randint(0, len(self.deck_cards[tier])))

        for i in idx:
            cards.append(self.deck_cards[tier].pop(i))
        return cards

    def _check_nobles(self):
        # TODO:checks if any nobles can be given to current player
        # for noble in self.nobles:

        # if noble can be given, update player states, remove noble from board
        pass

    def _end_player(self):
        # check for win condition, return player
        if self.current_player.points >= self.points_to_win:
            print(f'Player {self.current_player.id} has won! {self.current_player.points} points!')
            return self.current_player
        else:
            if self.current_player.id == self.player2.id:
                self.turn += 1
            self.current_player = self.players.__next__()
            print(f'Round {self.turn}: Start')
