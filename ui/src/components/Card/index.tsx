import { FunctionComponent } from "react";
import { colorMapping } from "../../helpers/helpers";
import Token from "../Token";

export interface CardProps {
    id:number;
    cost:[number,number,number,number,number,number];
    color: "yellow"|"green"|'white'|'blue'|'black'|'red';
    point: number;
}
 
const Card: FunctionComponent<CardProps> = (props:CardProps) => {
    return (
        <div style={{
        border:`1px solid ${props.color}`,
        width:'128px',
        height:'192px',
        display:'flex'
        }}
        className="rounded-md flex-col"
        >
            <p className="text-xl pb-12 p-1">{props.point}</p>
            <div className="grid grid-cols-2 grid-row-auto gap-2 p-2">
                {
                    props.cost.map((cost,index)=> cost != 0 && (
                            <Token color={colorMapping[index]} cost={cost} key={index} />
                    ))
                }
            </div>
        </div>
    );
}
 
export default Card;