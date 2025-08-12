"use client";

import { KeyboardEvent, useEffect, useRef } from "react";
import Image from "next/image";
import { useAtom } from "jotai";
import { Message, SENDER } from "@/lib/types";
import { useSendQuery } from "@/hooks/useSendQuery";
import { botThinkingAtom, messageAtom } from "@/lib/state";
import Loader from "./loader";

const TextInput = () => {
  const [isBotThinking, setIsBotThinking] = useAtom(botThinkingAtom);
  const [msgs, setMsgs] = useAtom(messageAtom);
  const inputRef = useRef<HTMLInputElement>(null);
  const { mutateAsync: sendQueryAPI } = useSendQuery();

  useEffect(() => {
    scrollToBottom();
  }, [msgs]);

  const handleKeyUp = (ev: KeyboardEvent<HTMLInputElement>) => {
    if (ev.key.toLowerCase() !== "enter") return;
    sendQuery(inputRef.current?.value ?? "");
  };

  const handleClick = () => {
    sendQuery(inputRef.current?.value ?? "");
  };

  const sendQuery = async (query: string) => {
    if (!query) return;
    setIsBotThinking(true);
    const message: Message = {
      sender: SENDER.USER,
      content: query,
      timestamp: new Date(),
    };
    setMsgs((prev) => [...prev, message]);
    if (inputRef.current) inputRef.current.value = "";
    try {
      const response = await sendQueryAPI(message.content);
      const botMessage: Message = {
        sender: SENDER.BOT,
        content: response.reply,
        timestamp: new Date(),
      };
      setMsgs((prev) => [...prev, botMessage]);
      setIsBotThinking(false);
    } catch (error) {
      setIsBotThinking(false);
      console.error("Error in sending query: ", error);
    }
  };

  const scrollToBottom = () => {
    window.scrollTo({
      top: document.body.scrollHeight,
      behavior: "smooth",
    });
  };

  return (
    <>
      {isBotThinking && <Loader />}
      <div className="bg-gredient-dark flex w-full max-w-full flex-col">
        <div className="search-box mx-auto my-auto flex w-full flex-wrap items-center justify-between sm:w-full md:w-full lg:w-full xl:w-full">
          <div className="mt-7 flex w-full rounded-lg border  border-gray-400 bg-gray-100 focus:border-gray-900 active:bg-white">
            {/* Speech to Text feature */}
            {/* <div className="cursor-pointer px-3 pt-4 hover:bg-yellow-400">
            <Image src={"/_mic.png"} width={25} height={30} alt={"mic"} />
          </div> */}
            <div className="flex h-auto w-full">
              <input
                id="queryBox"
                ref={inputRef}
                className="text-grey-darker h-auto w-full rounded-lg border border-gray-100 bg-gray-100 active:bg-white p-4 text-lg font-medium text-gray-600 outline-none focus:border-transparent focus:ring-0 placeholder-gray-400"
                type="search"
                placeholder="Ask me anything..."
                onKeyUp={handleKeyUp}
                disabled={isBotThinking}
              />
              <button
                className="flex h-auto w-20 items-center justify-center rounded rounded-l-none  bg-black"
                onClick={handleClick}
              >
                <Image src={"/_send.png"} width={25} height={25} alt={"send"} />
              </button>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default TextInput;
