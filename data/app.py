from flask import Flask
from flask import request, jsonify
from flask_cors import CORS
from rules import *
from ai import *
app = Flask(__name__)
CORS(app)
bot = MinMaxBot()


class InvalidAPIUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        super().__init__()
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv


@app.errorhandler(InvalidAPIUsage)
def invalid_api_usage(e):
    return jsonify(e.to_dict()), e.status_code


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/newGame")
def start_game():
    bot.board.startGame()
    return bot.board.returnState()


@app.route("/state")
def get_state():
    return bot.board.returnState()


@app.route("/action", methods=['POST'])
def post_action():
    body = request.json
    print("post action received: ", body)
    try:
        if(body['action'] == 0):
            card = Card(body['card']['id'], [body['card']['tier'], body['card']['value'],
                                             body['card']['type'] + 1, *body['card']['cost']])
            bot.board.current_player.takeAction(
                body['action'], card=card)
        else:
            bot.board.current_player.takeAction(**body)
    except Exception as err:
        print(err)
        raise InvalidAPIUsage(str(err), status_code=400)

    bot.ai_move()

    return bot.board.returnState()
