import React, { useReducer } from "react";
import { addCardToPlayer, BoardState, GameStats, PlayerState, removeCardFromBoard, setNextPlayer, subtractTokens } from "./helpers";

interface GameState {
    player1:PlayerState;
    player2:PlayerState;
    player3?:PlayerState;
    player4?:PlayerState;
    board:BoardState;
    stats:GameStats;
} 

const initialState:GameState = {
    player1:{
        cards:[
            {
                id:0,
                cost:[0,1,2,3,5,3],
                color:'green',
                point:0
            },
            {
                id:1,
                cost:[0,1,2,3,5,3],
                color:'green',
                point:0
            },
            {
                id:2,
                cost:[0,1,2,3,5,3],
                color:'red',
                point:0
            },
            {
                id:3,
                cost:[0,1,2,3,5,3],
                color:'yellow',
                point:0
            }
        ],
        reservedCards:[],
        tokens:[0,1,1,1,1,1],
        points:0
    },
    player2:{
        cards:[
            {
                id:4,
                cost:[0,1,2,3,5,3],
                color:'green',
                point:0
            },
            {
                id:5,
                cost:[0,1,2,3,5,3],
                color:'green',
                point:0
            },
            {
                id:6,
                cost:[0,1,2,3,5,3],
                color:'red',
                point:0
            },
            {
                id:7,
                cost:[0,1,2,3,5,3],
                color:'yellow',
                point:0
            }
        ],
        reservedCards:[],
        tokens:[0,1,1,1,1,1],
        points:0
    },
    board:{
        decks:[[
            {
                id:8,
                cost:[0,1,2,3,5,3],
                color:'green',
                point:0
            },
            {
                id:9,
                cost:[0,1,2,3,5,3],
                color:'green',
                point:0
            },
            {
                id:10,
                cost:[0,1,2,3,5,3],
                color:'red',
                point:0
            },
            {
                id:11,
                cost:[0,1,2,3,5,3],
                color:'yellow',
                point:0
            }
        ],[
            {
                id:12,
                cost:[0,1,2,3,5,3],
                color:'green',
                point:0
            },
            {
                id:13,
                cost:[0,1,2,3,5,3],
                color:'green',
                point:0
            },
            {
                id:14,
                cost:[0,1,2,3,5,3],
                color:'red',
                point:0
            },
            {
                id:15,
                cost:[0,1,2,3,5,3],
                color:'yellow',
                point:0
            }
        ],[
            {
                id:16,
                cost:[0,1,2,3,5,3],
                color:'green',
                point:0
            },
            {
                id:17,
                cost:[0,1,2,3,5,3],
                color:'green',
                point:0
            },
            {
                id:18,
                cost:[0,1,2,3,5,3],
                color:'red',
                point:0
            },
            {
                id:19,
                cost:[0,1,2,3,5,3],
                color:'yellow',
                point:0
            }
        ]],
        nobles:[
            {   
                points:3,
                cards:[0,4,4,4,0,0]
            },
            {   
                points:3,
                cards:[0,4,4,4,0,0]
            },
            {   
                points:3,
                cards:[0,4,4,4,0,0]
            },
            {   
                points:3,
                cards:[0,4,4,4,0,0]
            }
        ],
        cards:[[
            {
                id:20,
                cost:[0,1,2,3,5,3],
                color:'green',
                point:0
            },
            {
                id:21,
                cost:[0,1,2,3,5,3],
                color:'green',
                point:0
            },
            {
                id:22,
                cost:[0,1,2,3,5,3],
                color:'red',
                point:0
            },
            {
                id:23,
                cost:[0,1,2,3,5,3],
                color:'yellow',
                point:0
            },
        ],[
            {
                id:24,
                cost:[0,1,2,3,5,3],
                color:'green',
                point:0
            },
            {
                id:25,
                cost:[0,1,2,3,5,3],
                color:'green',
                point:0
            },
            {
                id:26,
                cost:[0,1,2,3,5,3],
                color:'red',
                point:0
            },
            {
                id:27,
                cost:[0,1,2,3,5,3],
                color:'yellow',
                point:0
            },
        ],[
            {
                id:28,
                cost:[0,1,2,3,5,3],
                color:'green',
                point:0
            },
            {
                id:29,
                cost:[0,1,2,3,5,3],
                color:'green',
                point:0
            },
            {
                id:30,
                cost:[0,1,2,3,5,3],
                color:'red',
                point:0
            },
            {
                id:31,
                cost:[0,1,2,3,5,3],
                color:'yellow',
                point:0
            }
        ]],
        tokens:[0,1,2,3,5,3]
    },
    stats:{
        target:15,
        turn: 8,
        player: 0,
    }
}

function reducer(state:any, action:any) {
    switch (action.type) {
      case 'BUY':
        if(state.stats.player === 0){
            return {
                player1:addCardToPlayer(state.player1, action.card,"BUY"),
                player2:{...state.player2},
                board:removeCardFromBoard(state.board,action.card),
                stats:setNextPlayer(state.stats)
          }
        } else {
            return {
                player1:{...state.player1},
                player2:addCardToPlayer(state.player2, action.card,"BUY"),
                board:removeCardFromBoard(state.board,action.card),
                stats:setNextPlayer(state.stats)
            }
        }
      case 'HOLD':
        if(state.stats.player === 0){
            return {
                player1:addCardToPlayer(state.player1, action.card,"HOLD"),
                player2:{...state.player2},
                board:removeCardFromBoard(state.board,action.card),
                stats:setNextPlayer(state.stats)
          }
        } else {
            return {
                player1:{...state.player1},
                player2:addCardToPlayer(state.player2, action.card,"HOLD"),
                board:removeCardFromBoard(state.board,action.card),
                stats:setNextPlayer(state.stats)
            }
        }
      default:
        return state
    }
  }

export const GameContext = React.createContext<{state?:GameState,dispatch?:React.Dispatch<any>}>({})
export const useGameProvider = () => {
    const [state, dispatch] = useReducer(reducer, initialState);
    return {state, dispatch}
}