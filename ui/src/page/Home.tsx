import { FunctionComponent } from "react";
import BoardDeckRow from "../components/BoardDeckRow";
import Noble from "../components/Noble";
import Player from "../components/Player";
import Stats from "../components/Stats";
import Token from "../components/Token";
import { BoardState, colorMapping, PlayerState } from "../helpers/helpers";

interface HomeProps {
    
}

const fakeStats = {
    target:15,
    turn: 8
}
const fakeState:PlayerState = {
    cards:[
        {
            id:1,
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
            id:1,
            cost:[0,1,2,3,5,3],
            color:'red',
            point:0
        },
        {
            id:1,
            cost:[0,1,2,3,5,3],
            color:'yellow',
            point:0
        }
    ],
    reservedCards:[],
    tokens:[0,1,1,1,1,1],
    points:0
}

const fakeboardState:BoardState = {
    decks:[[
        {
            id:1,
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
            id:1,
            cost:[0,1,2,3,5,3],
            color:'red',
            point:0
        },
        {
            id:1,
            cost:[0,1,2,3,5,3],
            color:'yellow',
            point:0
        }
    ],[
        {
            id:1,
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
            id:1,
            cost:[0,1,2,3,5,3],
            color:'red',
            point:0
        },
        {
            id:1,
            cost:[0,1,2,3,5,3],
            color:'yellow',
            point:0
        }
    ],[
        {
            id:1,
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
            id:1,
            cost:[0,1,2,3,5,3],
            color:'red',
            point:0
        },
        {
            id:1,
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
            id:1,
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
            id:1,
            cost:[0,1,2,3,5,3],
            color:'red',
            point:0
        },
        {
            id:1,
            cost:[0,1,2,3,5,3],
            color:'yellow',
            point:0
        },
    ],[
        {
            id:1,
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
            id:1,
            cost:[0,1,2,3,5,3],
            color:'red',
            point:0
        },
        {
            id:1,
            cost:[0,1,2,3,5,3],
            color:'yellow',
            point:0
        },
    ],[
        {
            id:1,
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
            id:1,
            cost:[0,1,2,3,5,3],
            color:'red',
            point:0
        },
        {
            id:1,
            cost:[0,1,2,3,5,3],
            color:'yellow',
            point:0
        }
    ]],
    tokens:[0,1,2,3,5,3]
}
const Home: FunctionComponent<HomeProps> = () => {
    return (
        <div className="bg-slate-600 h-full min-h-screen flex">
            <div className="w-1/3 h-full flex flex-col">
                <Stats {...fakeStats} />
                <div className='flex-1 flex flex-col justify-between py-10 ml-32'>
                    <Player id="Player 1" state={fakeState} />
                    <Player id="Player 2" state={fakeState} />
                    <Player id="Player 3" state={fakeState} />
                    <Player id="Player 4" state={fakeState} />
                </div>
            </div>
            <div className="w-2/3 items-end h-full flex flex-col pl-40 mr-32">
                <div className="flex gap-10 pl-16 mb-10 mt-5 ml-32">
                    {
                        fakeboardState.tokens.map((number, index)=>(
                            <div className="flex gap-1 items-center" key={index}>
                                <p>{number}</p>
                                <Token color={colorMapping[index]} size='xl'  />
                            </div>
                        ))
                    }
                </div>
                <div className='flex justify-start gap-4 ml-[17rem]'>
                {fakeboardState.nobles.map((noble,index)=>(
                        <Noble {...noble} key={index} />
                    ))}

                </div>
                <div className='flex-1 flex flex-col justify-between py-10 ml-32 gap-10'>
                    {fakeboardState.cards.map((card,index)=>(
                        <BoardDeckRow cards={card} deck={fakeboardState.decks[index]} key={index} />
                    ))}
                </div>
            </div>

        </div>
    );
}
 
export default Home;