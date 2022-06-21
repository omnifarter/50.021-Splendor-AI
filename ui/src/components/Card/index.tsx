import { FunctionComponent, useContext, useState } from "react";
import { colorMapping } from "../../helpers/helpers";
import Token from "../Token";
import Modal from 'react-modal';
import { GameContext } from "../../helpers/useGameProvider";

export interface CardProps {
    id:number;
    cost:[number,number,number,number,number,number];
    color: "yellow"|"green"|'white'|'blue'|'black'|'red';
    point: number;
    forSale?:boolean; //passed in if the card is on the board.
}
 
const Card: FunctionComponent<CardProps> = (props:CardProps) => {

    const [open, setOpen] = useState(false)
    const {dispatch} = useContext(GameContext)

    const openModal = () => {
        setOpen(true)
    }
    
    const closeModal = () => {
        setOpen(false)
    }

    const onBuy = () => {
        dispatch && dispatch({type:"BUY",card:props})
        closeModal()
    }

    const onHold = () => {
        dispatch && dispatch({type:"HOLD",card:props})
        closeModal()
    }
    return (
        <>
        {/* @ts-ignore */}
        <Modal isOpen={open} onRequestClose={closeModal} ariaHideApp={false} style={
            {
                content:{
                    display:'flex',
                    color:'white',
                    backgroundColor:'#64748b',
                    border:'none',
                    boxShadow:'0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)',
                    height:'236px',
                    width:'300px',
                    top:"50%",
                    left:"50%",
                    transform:'translate(-50%, -50%)'
                },
                overlay:{
                    backgroundColor:'rgba(0,0,0,0)'
                }
            }
        }>            
            <button type="button" className="absolute right-4 text-gray-400 bg-transparent rounded-lg text-sm p-1.5 ml-auto inline-flex items-center" onClick={closeModal} >
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><path  d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"></path></svg>  
            </button>
            <div style={{
            border:`1px solid ${props.color}`,
            width:'128px',
            height:'192px',
            display:'flex'
            }}
            className="rounded-md flex-col"
            onClick={props.forSale ? openModal : undefined}
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
            <div className="flex flex-col justify-between py-6 ml-6">
                <button className='text-2xl' onClick={onBuy}>buy</button>
                <button className='text-2xl' onClick={onHold}>hold</button>
            </div>
        </Modal>
            <div style={{
            border:`1px solid ${props.color}`,
            width:'128px',
            height:'192px',
            display:'flex'
            }}
            className="rounded-md flex-col"
            onClick={props.forSale ? openModal : undefined}
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
        </>
    );
}
 
export default Card;