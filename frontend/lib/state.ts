import { atom } from "jotai";
import { Message } from "./types";

export const messageAtom = atom<Array<Message>>([
  // { sender: SENDER.USER, content: "Hi", timestamp: new Date() },
  // {
  //   sender: SENDER.BOT,
  //   content:
  //     "To help you find the right products, I'd need a bit more detail. Can you specify what you're looking for? For example:\n- A product name or keyword (e.g., \"products with 'Jet'\")\n- SKU or item number\n- Attributes like finish (e.g., \"matte\"), color, or size\n- Thickness or any specific feature\n\nLet me know how you'd like to refine your search! 😊",
  //   timestamp: new Date(),
  // },
  // { sender: SENDER.USER, content: "Who are you?", timestamp: new Date() },
]);

export const botThinkingAtom = atom<boolean>(false);
