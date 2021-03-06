import { FunctionComponent } from "react";


export interface TokenProps {
    color:string
    cost?:number
    size?:'xl'
}
 
const Token: FunctionComponent<TokenProps> = (props:TokenProps) => {
    if (props.cost){
        return (
            <div className="flex w-6">
                <p className={`relative w-0 left-2 top-0.5 text-sm ${props.color === 'white'&& "text-black" }`}>{props.cost}</p>
                <div style={{backgroundColor:props.color,borderRadius:'24px',content:""}} className="h-6 w-full" />
            </div>
            )
    }else{
        return (
    
            <div style={{backgroundColor:props.color,borderRadius:'24px',content:""}} className={`${ props.size ==='xl' ? 'h-12 w-12' :"h-6 w-6"}`} />
        );
    }
}
 
export default Token;