from flask import Flask
from flask import request

from rules import *
app = Flask(__name__)
app.config["DEBUG"] = True

board = Board()

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/newGame")
def start_game():
    board.startGame()
    return board.returnState()

@app.route("/state")
def get_state():
    return board.returnState()

@app.route("/action",methods=['POST'])
def post_action():
    body = request.json
    print("post action received: ",body)
    board.current_player.takeAction(**body)
    return board.returnState()