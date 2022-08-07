import { FunctionComponent } from "react";
import Card, { CardProps } from "../Card";

interface BoardDeckRowProps {
    cards:CardProps[]
    deck:CardProps[]
    onSubmitCardAction(card:any):void

}
 
const BoardDeckRow: FunctionComponent<BoardDeckRowProps> = (props:BoardDeckRowProps) => {
    return (
        <div className="flex items-center">
            <div className="border-slate-300 border-solid border-2 w-24 h-32 flex justify-center items-center mr-12">
                <p className="text-3xl">{props.deck.length}</p>
            </div>
            <div className="flex gap-4">
                {props.cards.map((card,index)=>(
                    <Card {...card} key={index} forSale={true} onClick={() => props.onSubmitCardAction(card)} />
                ))}
            </div>
        </div>
    );
}
 
export default BoardDeckRow;