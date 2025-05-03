from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama

class LLMService:
    def __init__(self):
        template = """You are a kind and friendly AI assistant. Respond only in Serbian, clearly and briefly (under 20 words).
The previous conversation was:
{history}
And the user now says: {input}
Your response:
"""
#         template = """
# Ti si ljubazan i prijateljski AI asistent. Odgovaraj isključivo na srpskom jeziku, jasno i kratko (manje od 20 reči).
# Prethodni razgovor je:
# {history}
# A korisnik sada kaže: {input}
# Tvoj odgovor:
# """
        prompt = PromptTemplate(input_variables=["history", "input"], template=template)
        self.chain = ConversationChain(
            prompt=prompt,
            memory=ConversationBufferMemory(ai_prefix="Assistant:"),
            llm=Ollama(model="tinyllama"),
            verbose=False,
        )

    def get_response(self, user_input: str) -> str:
        response = self.chain.predict(input=user_input)
        return response.removeprefix("Assistant:").strip()
