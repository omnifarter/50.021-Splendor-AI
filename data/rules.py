# For importing the game data (cards, tokens etc.) into their specified data structures
from enum import IntEnum
import math
import random

from typing import List
import numpy as np

card_path = './cards.csv'
nobles_path = './nobles.csv'

"""
Tokens are positive if player is taking, negative if player is returning.
""" 

class Colour(IntEnum):
    GREEN = 0
    WHITE = 1
    BLUE = 2
    BLACK = 3
    RED = 4
    GOLD = 5

class Action(IntEnum):
    BUY_CARD = 0
    BUY_RESERVE = 1
    TAKE_TOKEN = 2
    RESERVE_CARD = 3

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


class Noble:
    def __init__(self, id, cost):
        # cost: array of ints representing total card cost for each type required to buy the noble
        self.id = id
        self.cost = cost    
        self.points = 3

    def __str__(self):
        return f"ID: {self.id}, Cost: {self.cost}"


class Board:
    def __init__(self):
        # read all data
        self.all_cards, self.nobles = self._read_data()
        self.open_cards = [[],[],[]]
        self.deck_cards = [[],[],[]]

    def _read_data(self):
        # reads card and nobles into their respective class objects. Stores in array.
        temp_cards = np.genfromtxt(card_path, dtype=np.int32, delimiter=',', skip_header=1)
        temp_nobles = np.genfromtxt(nobles_path, dtype=np.int32, delimiter=',', skip_header=1)

        cards = []
        for i in range(len(temp_cards)):
            c = Card(i, temp_cards[i])
            cards.append(c)

        nobles = []
        for i in range(len(temp_nobles)):
            n = Noble(i, temp_nobles[i])
            nobles.append(n)

        return cards, nobles
    
    def reserveCard(self, card):
        row_index = -1
        card_index = -1
        for i, row in enumerate(self.open_cards):
            try:
                card_index = searchCardIndex(row, card)
                row_index = i
            except:
                continue
        if card_index == -1 or row_index == -1:
            raise Exception('BOARD_CARD_NOT_FOUND')

        self.open_cards[row_index].pop(card_index)
        self._openCard(row_index)

    def _openCard(self, row_index):
        next_card = self.deck_cards[row_index].pop()
        self.open_cards[row_index].append(next_card)
    
    def startGame(self):
        # fill deck cards
        for card in self.all_cards:
            self.deck_cards[card.tier - 1].append(card)
        
        # open 3 cards per row.
        for row_index in range(len(self.deck_cards)):
            for i in range(3):
                self._openCard(row_index)

        self.bank = TokenBank(2)
        self.player1 = PlayerState(id=0, turn_order=0,board=self,bank=self.bank)
        self.player2 = PlayerState(id=1,turn_order=1,board=self,bank=self.bank)
        self.currentPlayer = self.player1
        self.turn = 1
        self.points_to_win = 15
        print("Game started!")

    def endTurn(self, player):
        print("Player {} ended turn.",player.id)
        if self.player1.id == player.id:
            self.currentPlayer = self.player2
        elif self.player2.id == player.id:
            self.currentPlayer = self.player1
            self.turn += 1
            print("Round ended. Next round: {}".format(self.turn))
        else:
            raise Exception("BOARD_INVALID_PLAYER")
class TokenBank:
    def __init__(self, num_players):
        assert 2 <= num_players <= 4, "number of players should be between 2 and 4"
        starting_tokens = [4, 5, 7][num_players-2]
        self.tokens = [starting_tokens] * 5 + [5]

    def update(self, tokens):
        for i, t in enumerate(tokens):
            updatedCount = self.tokens[i] - t
            if updatedCount > 5:
                raise Exception('BANK_EXCEED_TOKENS')
            self.tokens[i] = updatedCount


class PlayerState:
    def __init__(self, id, turn_order, board: Board, bank: TokenBank):
        self.id = id
        self.turn_order = turn_order
        self.points = 0
        # Allows player to reference board and bank states
        self.board = board
        self.bank = bank

        # Player inventory
        self.cards = []
        self.card_counts = [0, 0, 0, 0, 0]
        self.reserved_cards = []
        self.tokens = [0, 0, 0, 0, 0, 0]
    
    def __str__(self):
        return "\nPlayer {}:\nPoints: {}\nTokens: {}\nCards: {}Reserves: {}".format(self.id,self.points,self.tokens,self.cards,self.reserved_cards)
    def takeAction(self, action:Action, **kwargs):
        if action == Action.BUY_CARD:
            self._updateTokens(kwargs['tokens'])
            self.takeCard(kwargs['card'])

        elif action == Action.BUY_RESERVE:
            self._updateTokens(kwargs['tokens'])
            self.buyReserve(kwargs['card'])

        elif action == Action.RESERVE_CARD:
            self._updateTokens(kwargs['tokens'])
            self.reserveCard(kwargs['card'])

        elif action == Action.TAKE_TOKEN:
            self._updateTokens(kwargs['tokens'])

        else:
            raise Exception('EMPTY_ACTION')

        self.board.endTurn(self)

    ## Define player actions
    def takeToken(self, tokens):
        # Player is allowed to draw 3 tokens of different colour, or 2 tokens of same colour,
        # provided there are 4 tokens of that colour in the bank
        multiToken = False
        for i, token in enumerate(tokens):
            if token > 2 or (token == 2 and multiToken):
                raise Exception('PLAYER_TAKING_TOO_MANY_TOKENS')
            elif token == 2:
                if self.bank.tokens[i] < 4:
                    raise Exception('BANK_LESS_THAN_4_TOKENS')
                multiToken = True
        self._updateTokens(tokens)


    def takeCard(self, card):
        self.cards.append(card)
        self.card_counts[card.type] += 1
        self.points += card.points

        if self.points >= self.board.points_to_win:
            print("PLAYER {} HAS WON!".format(self.id))
        

    def buyReserve(self, card):
        card_index = searchCardIndex(self.reserved_cards, card)
        self.takeCard(card)
        self.reserved_cards.pop(card_index)

    def reserveCard(self, card):
        # Player picks a card on the board or from top of deck to add to their reserve pile
        # Upon reserving a card, award player with 1 gold token
        self.board.reserveCard(card)
        self.tokens[5] += 1

    def _checkValidToken(self,a,b):
        for token, i in enumerate(b):
            if token < a[i]:
                return False
        return True

    def _updateTokens(self,tokens):
        if self._checkValidToken(self.tokens, tokens):
            self.tokens = [t + tokens[i] for i,t in enumerate(self.tokens)]
            self.bank.update(tokens)
        else:
            raise Exception('PLAYER_NOT_ENOUGH_TOKENS')


def searchCardIndex(cardList, card):
        card_index = -1
        for i, reserved_card in enumerate(cardList):
            if card.id == reserved_card.id:
                card_index = i
                break
        if card_index == -1:
            raise Exception('CARD_NOT_FOUND')
        return card_index
