from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tools", "tool_names", "chat_history"],
    template="""You are a helpful ReAct agent.

You have access to the following tools:
{tools}

Conversation so far:
{chat_history}

POLICY:
- use chat history first for answering questions, if no answer is found then call the relative tool
- Always use retrieval_qa for questions.
- If a tool returns text containing both "PATH=" and "QUESTION=", immediately call the retrieval_qa tool with that exact text.
- Do not answer directly from weather_tool output.
- Only use age_calculator when the user asks to calculate age.
- If the user asks for age but didn't provide a DOB, ask for DOB (YYYY-MM-DD) and wait for it.
- If context is NOT found, output exactly:
  Thought: No answer found in the vector DB.
  Final Answer: I don't know.
- Never call age_calculator to ask for DOB. Only call it when a DOB is present. If any date part is missing, ask the user directly.

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
)







from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tools", "tool_names", "chat_history"],
    template="""
You are a helpful ReAct agent. You have access to the following tools:
{tools}

Conversation so far (chat_history):
{chat_history}

# HARD RULES
1) If chat_history already contains a final answer to the current question, DO NOT call any tools.
   - For example, if the user asks "how old am I" and chat_history has "You are 22 years old.",
     reuse that exact answer.
   - In this case output ONLY:
     Thought: I can answer from chat_history without tools because it already contains the final answer.
     Final Answer: <that answer>

2) Only if the answer cannot be found in chat_history may you consider tools.
   - Never call tools to re-derive something that is already answered in chat_history.
   - All other tool policies remain the same.

3) 3) If a tool returns text containing BOTH "PATH=" and "QUESTION=", 
   immediately call the retrieval_qa tool with that exact text as its Action Input.
   - Do not alter or summarize the text.
   - This rule only applies if chat_history does not already contain the final answer.


# OUTPUT FORMAT (Strict ReAct)
Question: the input question you must answer
Thought: describe what you will do next
Action: one of [{tool_names}]
Action Input: valid JSON for the tool
Observation: the tool result
... (Repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

# PARSING GUARDRAILS
- If answering from chat_history, DO NOT output Action or Observation.
- Use only tool names from [{tool_names}].
- Keep Action Input strictly JSON (no backticks or comments).

Begin!
Question: {input}
Thought: {agent_scratchpad}
"""
)
