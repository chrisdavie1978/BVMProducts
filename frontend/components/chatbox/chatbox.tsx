"use client";

import TextInput from "./textInput";
import Conversation from "./conversation/conversation";
import ReactQueryClientProvider from "@/providers/ReactQueryClientProvider";

const ChatBox = () => {
  return (
    <ReactQueryClientProvider>
      <div className="h-auto w-full mt-auto">
        <div className="flex w-full flex-col px-5 pb-20 text-neutral-700 max-md:mt-10 max-md:max-w-full text-center ">
          <Conversation />
          <TextInput />
        </div>
      </div>
    </ReactQueryClientProvider>
  );
};

export default ChatBox;
