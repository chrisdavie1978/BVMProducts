import { useEffect } from "react";
import { useAtom } from "jotai";
import Message from "./message";
import { getFormattedTimestamp } from "@/lib/utils/timestampFormatter";
import { useSendQuery } from "@/hooks/useSendQuery";
import { Message as MessageType, SENDER } from "@/lib/types";
import { botThinkingAtom, messageAtom } from "@/lib/state";

const Conversation = () => {
  const [msgs, setMsgs] = useAtom(messageAtom);
  const [, setIsBotThinking] = useAtom(botThinkingAtom);
  const { mutateAsync: sendQueryAPI } = useSendQuery();

  useEffect(() => {
    msgs.sort((a, b) => {
      return a.timestamp.getTime() - b.timestamp.getTime();
    });
  }, [msgs]);

  useEffect(() => {
    (async () => {
      try {
        let sessionId = sessionStorage.getItem("chatSessionId");
        if (!sessionId) {
          sessionId = Date.now().toString();
          sessionStorage.setItem("chatSessionId", sessionId);
        }
        setIsBotThinking(true);
        const response = await sendQueryAPI("Introduce yourself");
        const botMessage: MessageType = {
          sender: SENDER.BOT,
          content: response.reply,
          timestamp: new Date(),
        };
        setMsgs((prev) => [...prev, botMessage]);
        setIsBotThinking(false);
      } catch (error) {
        console.error("Error in sending query: ", error);
        setIsBotThinking(false);
      }
    })();
  }, []);

  return (
    <>
      {msgs.map((msg) => (
        <div key={getFormattedTimestamp(msg.timestamp)}>
          <Message message={msg} />
        </div>
      ))}
    </>
  );
};

export default Conversation;
