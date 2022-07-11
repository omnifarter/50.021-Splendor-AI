import { FunctionComponent, useContext } from "react";
import { useQuery } from "react-query";
import BoardDeckRow from "../components/BoardDeckRow";
import { CardProps } from "../components/Card";
import Noble, { NobleProps } from "../components/Noble";
import Player from "../components/Player";
import Stats from "../components/Stats";
import Token from "../components/Token";
import { colorMapping, PLAYERS } from "../helpers/helpers";
import {newGame} from '../helpers/my_apis'

const Home: FunctionComponent = () => {
    const {isLoading , error, data} : {isLoading:boolean, data:any, error:any} =  useQuery('state', newGame)

    return (
        <>
        {error && <p>error!</p>}
        {isLoading && <p>Loading...</p>}
        {data &&
        <div className="bg-slate-600 h-full min-h-screen flex">
            <div className="w-1/3 h-full flex flex-col">
                <Stats target={15} player={data.current_player.id} turn={data.turn} />
                <div className='flex-1 flex flex-col justify-between py-10 ml-32'>
                    <Player id={data.player1.id} state={{...data.player1}} />
                    <Player id={data.player2.id} state={{...data.player2}}/> 
                </div>
            </div>
            <div className="w-2/3 items-end h-full flex flex-col pl-40 mr-32">
                <div className="flex gap-10 pl-16 mb-10 mt-5 ml-32">
                    {
                        data.bank.tokens.map((number:number, index:number)=>(
                            <div className="flex gap-1 items-center" key={index}>
                                <p>{number}</p>
                                <Token color={colorMapping[index]} size='xl'  />
                            </div>
                        ))
                    }
                </div>
                <div className='flex justify-start gap-4 ml-[17rem]'>
                {data.nobles.map((noble:NobleProps,index:number)=>(
                        <Noble {...noble} key={index} />
                    ))}

                </div>
                <div className='flex-1 flex flex-col justify-between py-10 ml-32 gap-10'>
                    {data.open_cards.map((card:CardProps[],index:number)=>(
                        <BoardDeckRow cards={card} deck={data.deck_cards[index]} key={index} />
                    ))}
                </div>
            </div>
        </div>}
        </>
    );
}
 
export default Home;