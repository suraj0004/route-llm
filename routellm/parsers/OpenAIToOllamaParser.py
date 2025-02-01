from datetime import datetime
import logging
import json
class OpenAIToOllamaParser:
    def __init__(self):
        pass
    def parse_chunk(self, chunk):
            """
            Parses a chunk from OpenAI's response stream and converts it to Ollama's response format.
            This method yields the converted chunks in Ollama's format.
            """
            # logging.info('\n chunk_____________________\n')
            # logging.info(chunk)

            # Split the response by the new line separator to get individual chunks
            chunks = chunk.split(b'\n\n')
            model = ''
            # Iterate over each chunk and parse it
            for chunk in chunks:
                if chunk:
                    # If the chunk is [DONE], it indicates the end of the response
                    if chunk.lstrip(b"data: ") == b"[DONE]":
                        logging.info("Received [DONE] - ending response")
                        ollama_chunk = {
                            "created": int(datetime.now().timestamp()),
                            "model": model,
                            "message": {
                                "role": "assistant",
                                "content": ""
                            },
                            "done": True
                        }

                        # Convert the Ollama chunk to the correct byte format with "data: " prefix
                        chunk_str = json.dumps(ollama_chunk)
                        chunk_data = f"{chunk_str}\n".encode("utf-8")

                        # Yield the Ollama formatted chunk
                        logging.info(f"Ollama formatted chunk:")
                        yield chunk_data
                        break  # End the streaming on [DONE]
                    else:
                        # Remove the "data: " prefix and parse the chunk as JSON
                        chunk_data = chunk.lstrip(b"data: ")
                        try:
                            parsed_data = json.loads(chunk_data)
                            logging.info(f"Parsed data")

                            # Check if 'delta' is not empty and contains 'content'
                            delta = parsed_data['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            model = parsed_data.get("model", "unknown")
                            # Check if 'finish_reason' is present and not None
                            finish_reason = parsed_data['choices'][0].get('finish_reason', None)
                            done = False
                            if finish_reason is not None:
                                done = True  # If finish_reason is not None, mark as done

                            # If content is available, prepare the chunk in Ollama format
                            if content:
                                ollama_chunk = {
                                    "created": int(datetime.now().timestamp()),
                                    "model": parsed_data.get("model", "unknown"),
                                    "message": {
                                        "role": "assistant",
                                        "content": content
                                    },
                                    "done": done
                                }

                                # Convert the Ollama chunk to the correct byte format with "data: " prefix
                                chunk_str = json.dumps(ollama_chunk)
                                chunk_data = f"{chunk_str}\n".encode("utf-8")

                                # Yield the Ollama formatted chunk
                                logging.info(f"Ollama formatted chunk")
                                yield chunk_data

                        except json.JSONDecodeError as e:
                            logging.error(f"Error decoding chunk: {e}")
                            continue


    def parse(self, openai_response):
        self.openai_response = openai_response
        logging.info('OpenAIToOllamaParser parse')
        ollama_response = {
            "model": self.openai_response.get("model"),
            "created_at": self._convert_to_iso8601(self.openai_response.get("created")),
            "message": self._extract_message(),
            "done_reason": self._extract_done_reason(),
            "done": True,
            "total_duration": self._mock_duration("total_duration"),
            "load_duration": self._mock_duration("load_duration"),
            "prompt_eval_count": self._mock_count("prompt_eval_count"),
            "prompt_eval_duration": self._mock_duration("prompt_eval_duration"),
            "eval_count": self._mock_count("eval_count"),
            "eval_duration": self._mock_duration("eval_duration"),
        }
        logging.info(f" \n\n parsed from openai to ollama: \n\n {json.dumps(ollama_response, indent=2)}  \n\n ")
        return ollama_response

    def _convert_to_iso8601(self, timestamp):
        """
        Converts a UNIX timestamp to ISO 8601 format.
        """
        if timestamp:
            return datetime.utcfromtimestamp(timestamp).isoformat() + "Z"
        return None
    

    def _extract_message(self):
        """
        Extracts the assistant's message content and parses tool_calls arguments.
        """
        choices = self.openai_response.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            result = {
                "role": "assistant",
                "content": message.get("content", "")
            }
            
            if "tool_calls" in message:
                tool_calls = message["tool_calls"]
                for tool in tool_calls:
                    if "function" in tool and "arguments" in tool["function"]:
                        try:
                            # Parse arguments string to a Python dictionary
                            tool["function"]["arguments"] = json.loads(tool["function"]["arguments"])

                        except json.JSONDecodeError:
                            # Log error and leave arguments as is if parsing fails
                            tool["function"]["arguments"] = {}

                result["tool_calls"] = tool_calls
                result["content"] = json.dumps(tool_calls, indent=2)

            return result
        return {"role": "assistant", "content": ""}



    def _extract_done_reason(self):
        """
        Extracts the reason why the operation was completed.
        """
        choices = self.openai_response.get("choices", [])
        if choices:
            return choices[0].get("finish_reason", "unknown")
        return "unknown"

    def _mock_duration(self, key):
        """
        Mocked duration for testing purposes. Replace with actual logic if available.
        """
        return 89743835771 if key == "total_duration" else 103644548

    def _mock_count(self, key):
        """
        Mocked count for testing purposes. Replace with actual logic if available.
        """
        return 168 if key == "eval_count" else 11