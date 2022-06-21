import { FunctionComponent, useContext } from "react";
import BoardDeckRow from "../components/BoardDeckRow";
import { CardProps } from "../components/Card";
import Noble, { NobleProps } from "../components/Noble";
import Player from "../components/Player";
import Stats from "../components/Stats";
import Token from "../components/Token";
import { colorMapping, PLAYERS } from "../helpers/helpers";
import { GameContext } from "../helpers/useGameProvider";


const Home: FunctionComponent = () => {
    const {state} = useContext(GameContext)

    return (
        <>
        {state &&
        <div className="bg-slate-600 h-full min-h-screen flex">
            <div className="w-1/3 h-full flex flex-col">
                <Stats {...state.stats} />
                <div className='flex-1 flex flex-col justify-between py-10 ml-32'>
                   
                   { //@ts-ignore
                    PLAYERS.map((id)=><Player id={id} state={state[id]} />)
                   }
                </div>
            </div>
            <div className="w-2/3 items-end h-full flex flex-col pl-40 mr-32">
                <div className="flex gap-10 pl-16 mb-10 mt-5 ml-32">
                    {
                        state.board.tokens.map((number:number, index:number)=>(
                            <div className="flex gap-1 items-center" key={index}>
                                <p>{number}</p>
                                <Token color={colorMapping[index]} size='xl'  />
                            </div>
                        ))
                    }
                </div>
                <div className='flex justify-start gap-4 ml-[17rem]'>
                {state.board.nobles.map((noble:NobleProps,index:number)=>(
                        <Noble {...noble} key={index} />
                    ))}

                </div>
                <div className='flex-1 flex flex-col justify-between py-10 ml-32 gap-10'>
                    {state.board.cards.map((card:CardProps[],index:number)=>(
                        <BoardDeckRow cards={card} deck={state.board.decks[index]} key={index} />
                    ))}
                </div>
            </div>
        </div>}
        </>
    );
}
 
export default Home;