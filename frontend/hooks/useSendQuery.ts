import { useMutation, UseMutationResult } from "@tanstack/react-query";

export type SendQueryResponse = {
  reply: string;
};

export const useSendQuery = (): UseMutationResult<
  SendQueryResponse,
  Error,
  string
> => {
  return useMutation<SendQueryResponse, Error, string>({
    mutationKey: ["query"],
    mutationFn: async (query: string) => {
      const payload = {
        message: query,
        session_id: sessionStorage.getItem("chatSessionId"),
      };

      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        if (!response.ok) {
          const errorMessage = await response.text();
          throw new Error(errorMessage);
        }

        const responseJSON: SendQueryResponse = await response.json();
        return responseJSON;
      } catch (error) {
        console.error("Send Query Hook:", error);
        throw error;
      }
    },
  });
};
