import json
import logging
class OllamaRequestParser:
    def __init__(self):
        self.endpoint_mapping = {
            "api/generate": {
                "openai": "https://api.openai.com/v1/completions",
                "claude": "https://api.anthropic.com/v1/completions",
                "groq": "https://api.groq.com/v1/completions"
            },
            "api/chat": {
                "openai": "https://api.openai.com/v1/chat/completions",
                "claude": "https://api.anthropic.com/v1/chat/completions",
                "groq": "https://api.groq.com/v1/chat/completions"
            },
        }

    def parse_request(self, ollama_url, ollama_body, output_type):

        if output_type not in ["openai", "claude", "groq"]:
            raise ValueError(f"Unsupported output type: {output_type}")

        if ollama_url not in self.endpoint_mapping:
            raise ValueError(f"Unsupported Ollama endpoint: {ollama_url}")

        api_url = self.endpoint_mapping[ollama_url].get(output_type)
        if not api_url:
            raise ValueError(f"No URL mapping found for {output_type} at {ollama_url}")

        body = self._map_body(ollama_url, ollama_body, output_type)

        return {
            "url": api_url,
            "body": body
        }

#v3
    def _parse_body_messages(self, messages):
        """
        Parses messages to extract tool calls and update tool_call_id in tool messages.
        """
        if not messages or messages[-1]["role"] != "tool":
            return messages  # No processing needed if messages list is empty

        # Initialize variables to store results
        tool_meta = []
        tool_mapping = {}

        # Initialize list to store assistant message indices
        assistant_indices = []
        
        # Loop through messages and find assistant messages followed by tool messages
        for i in range(len(messages) - 1):
            if messages[i]["role"] == "assistant" and messages[i + 1]["role"] == "tool":
                assistant_indices.append(i)

        if not assistant_indices:
            return messages  # No assistant message followed by a tool message

        # Process each set of tool calls
        for assistant_index in assistant_indices:
            # Find the range of tool messages following this assistant message
            tool_start_index = assistant_index + 1
            tool_end_index = len(messages) - 1

            # Parse tool metadata from the assistant's message
            try:
                tool_meta = json.loads(messages[assistant_index]["content"])

                # Convert function.arguments back to JSON string
                for tool in tool_meta:
                    tool["function"]["arguments"] = json.dumps(tool["function"]["arguments"])

                messages[assistant_index]["tool_calls"] = tool_meta  # Assign updated tool_meta to assistant message

                # Create a mapping of tool names to their corresponding tool IDs
                tool_mapping = {tool["function"]["name"]: tool["id"] for tool in tool_meta}

            except json.JSONDecodeError:
                continue  # Skip if JSON parsing fails for the assistant's message

            # Process tool messages and add tool_call_id
            for index in range(tool_start_index, tool_end_index + 1):

                if messages[index]["role"]  == "assistant":
                    break
                try:
                    tool_data = json.loads(messages[index]["content"])
                    tool_name = tool_data.get("tool_name")
                    tool_id = tool_mapping.get(tool_name)

                    if tool_id:
                        messages[index]["tool_call_id"] = tool_id  # Add tool_call_id to message

                except json.JSONDecodeError:
                    continue  # Skip malformed tool messages

        return messages


#v2
    # def _parse_body_messages(self, messages):
    #     """
    #     Parses messages to extract tool calls and update tool_call_id in tool messages.
    #     """
        # if not messages or messages[-1]["role"] != "tool":
        #     return messages  # No processing needed if the last message isn't a tool message

    #     # Find the last "assistant" message before the consecutive "tool" messages
    #     terminated_index = -1
    #     for i in range(len(messages) - 1, -1, -1):
    #         if messages[i]["role"] == "assistant":
    #             terminated_index = i
    #             break

    #     if terminated_index == -1:
    #         return messages  # No assistant message found before tool messages

    #     tool_start_index = terminated_index + 1
    #     tool_end_index = len(messages) - 1
    #     assistant_index = terminated_index

    #     # Parse tool metadata from the assistant's message
    #     tool_meta = []
    #     try:
    #         tool_meta = json.loads(messages[assistant_index]["content"])

    #         # Convert function.arguments back to JSON string
    #         for tool in tool_meta:
    #             tool["function"]["arguments"] = json.dumps(tool["function"]["arguments"])

    #         messages[assistant_index]["tool_calls"] = tool_meta  # Assign updated tool_meta to assistant message

    #     except json.JSONDecodeError:
    #         return messages  # Return original if JSON parsing fails

    #     # Create a mapping of tool names to their corresponding tool IDs
    #     tool_mapping = {tool["function"]["name"]: tool["id"] for tool in tool_meta}

    #     # Process tool messages and add tool_call_id
    #     for index in range(tool_start_index, tool_end_index + 1):
    #         try:
    #             tool_data = json.loads(messages[index]["content"])
    #             tool_name = tool_data.get("tool_name")
    #             tool_id = tool_mapping.get(tool_name)

    #             if tool_id:
    #                 messages[index]["tool_call_id"] = tool_id  # Add tool_call_id to message

    #         except json.JSONDecodeError:
    #             continue  # Skip malformed tool messages

    #     return messages


# v1
    # def _parse_body_messages(self, messages):
    #     """
    #     Parses messages to extract tool calls and update tool_call_id in tool messages.
    #     """
    #     if not messages or messages[-1]["role"] != "tool":
    #         return messages  # No processing needed if the last message isn't a tool message

    #     terminated_index = -1
    #     for i in range(len(messages) - 1, -1, -1):
    #         if messages[i]["role"] == "assistant":
    #             terminated_index = i
    #             break

    #     if terminated_index == -1:
    #         return messages  # No assistant message found before tool messages

    #     tool_start_index = terminated_index + 1
    #     tool_end_index = len(messages) - 1
    #     assistant_index = terminated_index

    #     # Parse tool metadata from the assistant's message
    #     try:
    #         tool_meta = json.loads(messages[assistant_index]["content"])
    #     except json.JSONDecodeError:
    #         return messages  # Return original if JSON parsing fails

    #     # Process tool messages and add tool_call_id
    #     for index in range(tool_start_index, tool_end_index + 1):
    #         try:
    #             tool_data = json.loads(messages[index]["content"])
    #             tool_name = tool_data.get("tool_name")
    #             tool_id = next((t["tool_id"] for t in tool_meta if t["tool_name"] == tool_name), None)

    #             if tool_id:
    #                 messages[index]["tool_call_id"] = tool_id  # Add tool_call_id to message

    #         except json.JSONDecodeError:
    #             continue  # Skip malformed tool messages

    #     logging.info(f" \n\n _parse_body_messages - ollama to openai: \n\n {json.dumps(messages, indent=2)}  \n\n ")
    #     return messages
        

    def _map_body(self, ollama_url, ollama_body, output_type):
        """
        Maps the Ollama body to the corresponding structure for the given output type.
        """
        mapped_body = {}

        # Map the common and specific fields for "api/chat"
        if ollama_url == "api/chat":
            mapped_body["model"] = ollama_body.get("model")
            mapped_body["messages"] = self._parse_body_messages(ollama_body.get("messages", [])) 
            mapped_body["tools"] = ollama_body.get("tools", []) if "tools" in ollama_body else None
            # mapped_body["format"] = ollama_body.get("format", "json")
            mapped_body["stream"] = ollama_body.get("stream", True)
            # mapped_body["keep_alive"] = ollama_body.get("keep_alive", "5m")
            # mapped_body["options"] = ollama_body.get("options", {})

            # Provider-specific adjustments
            if output_type == "openai":
                mapped_body["temperature"] = ollama_body.get("options", {}).get("temperature", 0.7)
            elif output_type == "claude":
                mapped_body["query"] = ollama_body.get("messages")
            elif output_type == "groq":
                mapped_body["text_input"] = ollama_body.get("messages")

        # Map the common and specific fields for "api/generate"
        elif ollama_url == "api/generate":
            mapped_body["model"] = ollama_body.get("model")
            mapped_body["prompt"] = ollama_body.get("prompt")
            # mapped_body["suffix"] = ollama_body.get("suffix", "")
            # mapped_body["images"] = ollama_body.get("images", [])
            # mapped_body["format"] = ollama_body.get("format", "json")
            mapped_body["stream"] = ollama_body.get("stream", True)
            # mapped_body["keep_alive"] = ollama_body.get("keep_alive", "5m")
            # mapped_body["options"] = ollama_body.get("options", {})
            # mapped_body["system"] = ollama_body.get("system", "")
            # mapped_body["template"] = ollama_body.get("template", "")

            # Provider-specific adjustments
            if output_type == "openai":
                mapped_body["temperature"] = ollama_body.get("options", {}).get("temperature", 0.7)
            elif output_type == "claude":
                mapped_body["query"] = ollama_body.get("prompt")
            elif output_type == "groq":
                mapped_body["text_input"] = ollama_body.get("prompt")

        return {k: v for k, v in mapped_body.items() if v is not None}
