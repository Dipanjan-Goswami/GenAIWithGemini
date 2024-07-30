from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_google_genai import GoogleGenerativeAI

from LoadProperties import LoadProperties

properties = LoadProperties()

generation_config = {
    "temperature": 1,
    "top_p": 0.30,
    "top_k": 10,
    "max_output_tokens": 200,
    "response_mime_type": "text/plain",
}

llm = GoogleGenerativeAI(model=properties.getModelName(), google_api_key=properties.getAPIkey(), generation_config =  generation_config)

## Case 1: Direct Invocation

response = llm.invoke("Tell me one fact about space")
print("Case1 Response - > ")
print(response)


## case 2: Invocation Via simple prompt input

template = """You are a chatbot having a conversation with a human.
Human: {human_input}
:"""

prompt = PromptTemplate(input_variables=["human_input"], template=template)
# prompt_val = prompt.invoke({"human_input":"Tell us in a exciting tone about Mumbai"})
# print("Prompt String is ->")
# print(prompt_val.to_string())

chain = prompt | llm

response = chain.invoke({"human_input":"Tell us in a exciting tone about Mumbai"})
print("Case2 Response - > ")
print(response)

## Case 3: Getting Response from Chat prompt with from_message function

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a chatbot that explains in steps. Do not disclose and private any Personal or private information"),
        ("ai", "I shall explain in steps"),
        ("user", "{input}"),
    ]
)

chain = prompt | llm
response = chain.invoke({"input": "What's the New York culture like?"})
print("Case3 Response - > ")
print(response)

## Case 4: getting Response using chatpromptTemplate from_template method

prompt = ChatPromptTemplate.from_template("Tell me a joke about {animal}")
chain1 = prompt | llm
response = chain1.invoke({"animal": "pig"})
print("Case4 Response - > ")
print(response)

## Case 5: Getting response using SystemMessage and HumanMessagePromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                "You are a helpful assistant that re-writes the user's text to "
                "sound more upbeat."
            )
        ),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
)

chain2 = prompt | llm
response = chain2.invoke({"text":"I don't like eating tasty things"})
print("Case5 Response ->")
print(response)