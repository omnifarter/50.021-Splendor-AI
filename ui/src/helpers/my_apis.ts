import { Action } from "./helpers"
import { showNotification } from '@mantine/notifications';
const BACKEND_URL = "http://127.0.0.1:5000"

export async function newGame(){
    console.log('calling new game...')
    const response = await fetch(BACKEND_URL + "/newGame")
    console.log('this is response',response)
    const data = await response.json()
    console.log(data)
    return data
}

export async function getState(){
    return await(await ( fetch(BACKEND_URL + "/state"))).json()
}

export async function getTokens(tokens:any){
    const res = await fetch(BACKEND_URL + '/action',{
        method:"POST",
        body:JSON.stringify({
            action: Action.TAKE_TOKEN,
            tokens
        }),
        headers:{
            'Content-Type':'application/json'
        }
    })
    if (res.status === 400) {
        
        showNotification({
            color:'red',
            message: (await res.json()).message
        })

    } else {
        return await res.json()
    }
}


export async function buyCard(card:any,){
    const res = await fetch(BACKEND_URL + '/action',{
        method:"POST",
        body:JSON.stringify({
            action: Action.BUY_CARD,
            card
        }),
        headers:{
            'Content-Type':'application/json'
        }
    })
    if (res.status === 400) {
        showNotification({
            color:'red',
            message: (await res.json()).message
        })
    } else {
        const state =  await res.json()
        return state
    }
    
}
