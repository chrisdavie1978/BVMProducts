export enum SENDER {
  USER = "user",
  BOT = "bot",
}

export type Message = {
  sender: SENDER;
  content: string;
  timestamp: Date;
};
