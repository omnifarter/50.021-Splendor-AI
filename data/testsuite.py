from lib2to3.pgen2 import token
from secrets import token_urlsafe
import unittest
from rules import *

class TestBoard(unittest.TestCase):

    board = None

    def setUp(self):
        self.board = Board()
        self.board.startGame()

    def tearDown(self):
        self.board = None
    
    def test_read_data_cards(self):
        var1, var2 = self.board._read_data()
        self.assertEqual(len(var1), 90)
        
    def test_read_data_nobles(self):
        var1, var2 = self.board._read_data()
        self.assertEqual(len(var2), 10)

    def test_starting_player(self):
        self.assertEqual(self.board.current_player, self.board.player1)

class TestTokens(unittest.TestCase):
    board = None
    def setUp(self):
        self.board = Board()
        self.board.startGame()

    def tearDown(self):
        self.board = None

class TestActions(unittest.TestCase):

    board = None

    def setUp(self):
        self.board = Board()
        self.board.startGame()

    def tearDown(self):
        self.board = None

    def test_take_tokens(self):
        self.board.current_player.takeAction(Action.TAKE_TOKEN,tokens=[0,2,0,0,0,0])
        print(self.board.player1)
        self.assertEqual(self.board.player1.tokens, [0,2,0,0,0,0])

    def test_bank_token_stock(self):
        self.board.current_player.takeAction(Action.TAKE_TOKEN,tokens=[0,2,0,0,0,0])
        self.assertEqual(self.board.bank.tokens, [4,2,4,4,4,5])

    def test_legal_take_multi(self):
        self.board.current_player.takeAction(Action.TAKE_TOKEN,tokens=[0,2,0,0,0,0])
        self.assertEqual(self.board.bank.tokens, [4,2,4,4,4,5])
        self.assertEqual(self.board.current_player, self.board.player2)
        self.board.current_player.takeAction(Action.TAKE_TOKEN,tokens=[1,1,1,0,0,0])
        self.assertEqual(self.board.current_player, self.board.player1)
        self.assertEqual(self.board.bank.tokens, [3,1,3,4,4,5])

    # Test Taking 2 tokens when token count is < 4
    # Player 1 takes 2, assert bank changes and player change. 
    # Player 2 attempts to take 2, bank must not change, player turn must not end
    def test_illegal_take_double(self):
        self.board.current_player.takeAction(Action.TAKE_TOKEN,tokens=[0,2,0,0,0,0])
        self.assertEqual(self.board.bank.tokens, [4,2,4,4,4,5])
        self.assertEqual(self.board.current_player.id, self.board.player2.id)

        with self.assertRaises(Exception) as context:
            self.board.current_player.takeAction(Action.TAKE_TOKEN,tokens=[0,2,0,0,0,0])
        self.assertTrue('BANK_LESS_THAN_4_TOKENS' in str(context.exception))
        self.assertEqual(self.board.current_player.id, self.board.player2.id)
        self.assertEqual(self.board.bank.tokens, [4,2,4,4,4,5])
    
        

if __name__ == '__main__':
    unittest.main()