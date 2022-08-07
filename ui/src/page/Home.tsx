import { FunctionComponent, useContext, useEffect, useState } from "react";
import { useMutation, useQuery } from "react-query";
import BoardDeckRow from "../components/BoardDeckRow";
import { CardProps } from "../components/Card";
import Noble, { NobleProps } from "../components/Noble";
import Player from "../components/Player";
import Stats from "../components/Stats";
import Token from "../components/Token";
import { colorMapping, PLAYERS } from "../helpers/helpers";
import { getTokens, getState, newGame, buyCard } from "../helpers/my_apis";
import _ from "lodash";
const Home: FunctionComponent = () => {
  const [gameState, setGameState] = useState<any | null>(null);
  const [bankTokens, setBankTokens] = useState(gameState?.bank?.tokens || []);
  const [tokens, setTokens] = useState([0, 0, 0, 0, 0, 0]);

  useEffect(() => {
    const asyncFn = async () => {
      const game = await newGame();
      setGameState(game);
    };
    asyncFn();
  }, []);

  const onSubmitTokenAction = async () => {
    resetTokens();
    const game = await getTokens(tokens);
    game && setGameState(game);
  };

  const onSubmitCardAction = async (card: any) => {
    const game = await buyCard(card);
    game && setGameState(game);
  };
  const onClickToken = (i: number) => {
    let tokensCopy = _.cloneDeep(tokens);
    tokensCopy[i] += 1;
    setTokens(tokensCopy);
    let bankTokensCopy = _.cloneDeep(bankTokens);
    bankTokensCopy[i] -= 1;
    setBankTokens(bankTokensCopy);
  };

  useEffect(() => {
    gameState && setBankTokens(gameState.bank.tokens);
  }, [gameState]);

  const resetTokens = () => {
    setBankTokens(gameState.bank.tokens);
    setTokens([0, 0, 0, 0, 0, 0]);
  };

  return (
    <>
      {gameState && (
        <div className="bg-slate-600 h-full min-h-screen flex">
          <div className="w-1/3 h-full flex flex-col">
            <Stats
              target={15}
              player={gameState.current_player.id}
              turn={gameState.turn}
            />
            <div className="flex-1 flex flex-col justify-between py-10 ml-32">
              <Player
                id={gameState.player1.id}
                state={{ ...gameState.player1 }}
              />
              <Player
                id={gameState.player2.id}
                state={{ ...gameState.player2 }}
              />
            </div>
          </div>
          <div className="w-2/3 items-end h-full flex flex-col pl-40 mr-32">
            <div className="flex gap-10 pl-16 mb-10 mt-5 ml-32">
              {bankTokens.map((number: number, index: number) => (
                <div>
                  <div className="flex gap-1 items-center" key={index}>
                    <p>{number}</p>
                    <Token
                      color={colorMapping[index]}
                      size="xl"
                      onClick={() => onClickToken(index)}
                    />
                  </div>
                  {index !== 5 && <p>{tokens[index]}</p>}
                </div>
              ))}
              <button className="text-2xl" onClick={resetTokens}>
                Reset Tokens
              </button>
              <button className="text-2xl" onClick={onSubmitTokenAction}>
                Get Tokens
              </button>
            </div>
            {/* <div className='flex justify-start gap-4 ml-[17rem]'>
                {gameState.nobles.map((noble:NobleProps,index:number)=>(
                        <Noble {...noble} key={index} />
                    ))}

                </div> */}
            <div className="flex-1 flex flex-col justify-between py-10 ml-32 gap-10">
              {gameState.open_cards.map((card: CardProps[], index: number) => (
                <BoardDeckRow
                  cards={card}
                  deck={gameState.deck_cards[index]}
                  key={index}
                  onSubmitCardAction={onSubmitCardAction}
                />
              ))}
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default Home;
