import { CSSProperties } from "react";
import { PulseLoader } from "react-spinners";
import { useAtom } from "jotai";
import { botThinkingAtom } from "@/lib/state";

const override: CSSProperties = {
  display: "inline-block",
};

const Loader = () => {
  const [isBotThinking] = useAtom(botThinkingAtom);
  return (
    <div className="h-10 text-left">
      <i className="text-gray">Bot is Thinking   </i>
      <PulseLoader
        color={"red"}
        loading={isBotThinking}
        size={6}
        cssOverride={override}
        aria-label="Loading Spinner"
        data-testid="loader"
      />
    </div>
  );
};

export default Loader;
