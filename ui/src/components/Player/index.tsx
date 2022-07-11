import { FunctionComponent, useEffect, useState } from "react";
import { PlayerState, colorMapping, colorIndexing } from "../../helpers/helpers";
import Card, { CardProps } from "../Card";
import PlayerCard from "../Card/PlayerCard";
import Token from "../Token";

interface PlayerProps {
    id: number;
    state:PlayerState
}
 
const Player: FunctionComponent<PlayerProps> = (props:PlayerProps) => {
    const [sortedCards, setSortedCards] = useState<CardProps[][]>()

    const sortCards = () => {
        const cards :CardProps[][] = [[],[],[],[],[],[]]
        
        props.state.cards.forEach((card)=>{
            cards[colorIndexing[card.type]].push(card)
        })

        setSortedCards(cards)
    }

    useEffect(()=>{
        sortCards()
    },[props.state.cards])
    return (
        <div style={{display:'flex',flexDirection:'column'}} className='p-4 rounded-md border-slate-400 border-solid border-2' >
            <div className="flex justify-between">
            <p className="mb-4 text-3xl">{props.id === 0 ? "Player" : "Splendor AI"}</p>
            <p className="text-3xl">{props.state.points}</p>
            </div>
            <div className="flex gap-3 mb-4 ">
                {
                    sortedCards && sortedCards.map((cards,index)=> (
                        cards.length != 0 && <PlayerCard cards={cards} color={colorMapping[index]} key={index}/>
                        ) 
                    )
                }
            </div>
            <div className="flex gap-3 mb-4 ">
                {
                    props.state.tokens.map((number, index)=>(
                        <div className="flex gap-1" key={index}>
                            {number}
                            <Token color={colorMapping[index]}  />
                        </div>
                    ))
                }
            </div>
            
        </div>
    );
}
 
export default Player;