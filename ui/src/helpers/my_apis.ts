const BACKEND_URL = "http://127.0.0.1:5000"

export async function newGame(){
    console.log('calling new game...')
    const response = await fetch(BACKEND_URL + "/newGame")
    console.log('this is response',response)
    const data = await response.json()
    console.log(data)
    return data
}

export function getState(){
    return fetch(BACKEND_URL + "/state")
}

