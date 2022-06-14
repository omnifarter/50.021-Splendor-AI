import { CardProps } from "../components/Card"
import { NobleProps } from "../components/Noble";

export const colorMapping:{[index:number]:string} = {
    0:'yellow',
    1:'green',
    2:'white',
    3:'blue',
    4:'black',
    5:'red'
}

export const colorIndexing:{[color:string]:number} = {
    'yellow':0,
    'green':1,
    'white':2,
    'blue':3,
    'black':4,
    'red':5
}

export interface PlayerState {
    cards:CardProps[];
    reservedCards:CardProps[];
    tokens:[number,number,number,number,number,number];
    points:number;
}

export interface BoardState {
    nobles:NobleProps[];
    cards:[CardProps[],CardProps[],CardProps[]];
    decks:[CardProps[],CardProps[],CardProps[]];
    tokens:[number,number,number,number,number,number];

}