import { FunctionComponent } from "react";
import { PLAYERS } from "../../helpers/helpers";

interface StatsProps {
    target: number;
    turn: number;
    player:number;
}
 
const Stats: FunctionComponent<StatsProps> = (props:StatsProps) => {
    return (
        <div className="flex w-1/2 justify-evenly mt-5">
            <p>Target: {props.target}</p>
            <p>Turn: {props.turn}</p>
            <p>Player: {PLAYERS[props.player]}</p>
        </div>
    );
}
 
export default Stats;