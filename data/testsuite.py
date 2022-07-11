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

    def setUp(self):
        board = Board()
        board.startGame()

    def tearDown(self):
        board = None


if __name__ == '__main__':
    unittest.main()