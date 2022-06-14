import { FunctionComponent, useState } from "react";
import Card, { CardProps } from ".";
import Modal from 'react-modal';

interface PlayerCardProps {
    color:string;
    cards:CardProps[];
}
 
const PlayerCard: FunctionComponent<PlayerCardProps> = (props: PlayerCardProps) => {
    const [open, setOpen] = useState(false)

    const openModal = () => {
        setOpen(true)
    }
    
    const closeModal = () => {
        setOpen(false)
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
                    height:'50%',
                    width:'50%',
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
            <div className="flex gap-4 flex-wrap p-10">
                {props.cards.map((card,index)=><Card {...card} key={index} />)}
            </div>
        </Modal>

        <div style={{        
        border:`1px solid ${props.color}`,
        width:'64px',
        height:'96px',
        display:'flex'
        }}
        className='text-center items-center rounded-md '
        onClick={openModal}
        >
            <p className="text-3xl w-full">{props.cards.length}</p>
        </div>
        </>
    );
}
 
export default PlayerCard;