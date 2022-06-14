import { FunctionComponent } from "react";
import { colorMapping } from "../../helpers/helpers";
import { CardProps } from "../Card";
import PlayerCard from "../Card/PlayerCard";

export interface NobleProps {
    points:number
    cards:[number,number,number,number,number,number]
}
 
const Noble: FunctionComponent<NobleProps> = (props:NobleProps) => {
    return (
        <div style={{
            border:`1px solid gold`,
            width:'128px',
            height:'192px',
        }}
        className='flex flex-col rounded-md justify-between'
        >
            <p className="text-xl p-1">{props.points}</p>
            <div className="flex flex-wrap gap-4 mb-4 ml-2">
                {props.cards.map((card,index)=>(
                    card !== 0 && <div style={{        
                        border:`1px solid ${colorMapping[index]}`,
                        width:'24px',
                        height:'36px',
                        display:'flex'
                        }}
                        className='text-center items-center rounded-md '
                        key={index}
                    >
                        <p className="text-sm w-full">{card}</p>
                    </div>
                    ))}
            </div>
        </div>
    );
}
 
export default Noble;