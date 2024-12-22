from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from llama_cpp import Llama

llm = Llama.from_pretrained(
	repo_id="TheBloke/Llama-2-7B-Chat-GGUF",
	filename="llama-2-7b-chat.Q2_K.gguf",
)



# Decode the output tokens back to text

from langchain.chains.prompt_selector import ConditionalPromptSelector
from langchain.prompts import PromptTemplate
DEFAULT_LLAMA_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""<<SYS>> \n You are an assistant tasked with improving Google search \
results. \n <</SYS>> \n\n [INST] Generate THREE Google search queries that \
are similar to this question. The output should be a numbered list of questions \
and each should have a question mark at the end: \n\n {question} [/INST]""",
)
DEFAULT_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an assistant tasked with improving Google search \
results. Generate THREE Google search queries that are similar to \
this question. The output should be a numbered list of questions and each \
should have a question mark at the end: {question}""",
)

QUESTION_PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=DEFAULT_SEARCH_PROMPT,
    conditionals=[(lambda llm: isinstance(llm, LlamaCpp), DEFAULT_LLAMA_SEARCH_PROMPT)],
)

prompt = QUESTION_PROMPT_SELECTOR.get_prompt(llm)

print(prompt)
from langchain.chains import LLMChain
llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "What is Taiwan known for?"
llm_chain.invoke({"question": question})

