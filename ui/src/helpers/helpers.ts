import { CardProps } from "../components/Card"
import { NobleProps } from "../components/Noble";
import _ from 'lodash'
export const PLAYERS = ['player1','player2']

export const colorMapping:{[index:number]:string} = {
    0:'yellow',
    1:'green',
    2:'white',
    3:'blue',
    4:'black',
    5:'red'
}

export const colorIndexing:{[color:string]:number} = {
    'yellow':0,
    'green':1,
    'white':2,
    'blue':3,
    'black':4,
    'red':5
}

export interface PlayerState {
    cards:CardProps[];
    reservedCards:CardProps[];
    tokens:[number,number,number,number,number,number];
    points:number;
}

export interface GameStats {
    target:number,
    turn:number,
    player:number
}
export interface BoardState {
    nobles:NobleProps[];
    cards:[CardProps[],CardProps[],CardProps[]];
    decks:[CardProps[],CardProps[],CardProps[]];
    tokens:[number,number,number,number,number,number];

}

export const subtractTokens = (
    player:[number,number,number,number,number,number],
    cost:[number,number,number,number,number,number]
    ) => {
    return player.map((val,ind)=>val-cost[ind])
}

export const removeCardFromBoard = (board:BoardState,card:CardProps) => {
    const rowIndex = findRowIndex(board,card)
    const copyBoard = _.cloneDeep(board)
    copyBoard.cards[rowIndex] = board.cards[rowIndex].filter((val)=>val.id !== card.id)
    const openedCard = copyBoard.decks[rowIndex].pop() 
    openedCard && copyBoard.cards[rowIndex].push(openedCard)
    return copyBoard
}

const findRowIndex = (board:BoardState,card:CardProps):number => {
    return board.cards.findIndex((row)=>row.some((c)=>c.id === card.id))
}
export const addCardToPlayer = (player:PlayerState, card:CardProps, type:"BUY"|"HOLD") => {
    return {
        cards:type == 'BUY' ? [...player.cards, card] : player.cards,
        reservedCards: type=='HOLD' ? [...player.reservedCards,card] : player.reservedCards,
        points: card.point ? player.points + card.point : player.points,
        tokens: subtractTokens(player.tokens,card.cost),
    }
}

export const setNextPlayer = (game:GameStats):GameStats => {
    if (game.player === PLAYERS.length - 1) {
        return {
            target:game.target,
            turn:game.turn + 1,
            player: 0
        }
    } else {
        return {
            target:game.target,
            turn:game.turn,
            player: game.player + 1
        }
    } 
}