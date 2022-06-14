import { FunctionComponent } from "react";

interface StatsProps {
    target: number;
    turn: number;
}
 
const Stats: FunctionComponent<StatsProps> = (props:StatsProps) => {
    return (
        <div className="flex w-1/2 justify-evenly mt-5">
            <p>Target: {props.target}</p>
            <p>Turn: {props.turn}</p>
        </div>
    );
}
 
export default Stats;